# Running TPS in parallel

Two ways:

| | `multi_run_` (one node) | study mode (`drivers/parallel.py`, any number of nodes and jobs) |
|---|---|---|
| Input | a list of (equilibrium, profile dict) | bouquet archives in a study dir (Cluster_planning/INTERFACE.md) |
| Scheduling | a `multiprocessing` pool, `imap_unordered` | claims on a shared-filesystem queue (`drivers/fsqueue.py`) |
| Restart | `done_{idx}.marker` (names its equilibrium) | per-draw status files; leases of dead workers are stolen |
| New inputs mid-run | no | yes (`feed`, also run by idle workers) |
| Output | `multi_compile_zarr` → one zarr | `compile` → one zarr per shot, incremental; `open_study` |

Other parallel helpers: `mre_recompute.recompute_mre_hyperparams` (chunks of runs in a pool),
`input_test_runner.run_multiple_scans_parallel` (scans across a pool), spot-checks (`multi_run_(spot_check=...)`).

## Environment knobs
- `TPS_NUM_WORKERS`: worker count for `multi_run_` (default: the SLURM allocation on the node).
- `TPS_WORK_ROOT`: node-local scratch for worker dirs (omega: `/tmp` or `$TMPDIR`, never NFS; Perlmutter: `/tmp`).
- `TPS_CODE_TIMEOUT`: seconds per RDCON/STRIDE/PEST3 run; the code's process group is killed after it (`TimeoutError`).
- `TPS_JGPEC_SLOTS`: at most this many concurrent jGPEC solves per node (`utils.node_slot`); unset: no limit.

## `multi_run_`
- One spawn pool on one node; each worker keeps a private working dir and log; dynamic scheduling.
- Per case: `result_{idx}.pkl`, `combined_xr_{idx}.nc` (tmp + rename), then `done_{idx}.marker` naming the
  equilibrium. Warm start skips a case whose marker names the same equilibrium (otherwise it is recomputed).
- Failures: `error_{idx}.json` as they happen, `errors.pkl` at the end. `fail_fast` stops at the first.
- Do not point two jobs at the same `master_working_dir`.

## Study mode

```bash
python -m tearing_physics_suite.drivers.parallel init    --study S --config cfg.json   # pins settings + build
python -m tearing_physics_suite.drivers.parallel feed    --study S                     # bouquet records -> items
python -m tearing_physics_suite.drivers.parallel work    --study S --workers W         # one per node (srun)
python -m tearing_physics_suite.drivers.parallel compile --study S [--finalize]        # per-shot zarr stores
python -m tearing_physics_suite.drivers.parallel status  --study S [-v] [--json f]
python -m tearing_physics_suite.drivers.parallel stop|drain|undrain --study S
```

`cfg.json` keys (`study.DEFAULTS`): `tps_kwargs` (for `nonlinear_resistive_calculation`), `variants`, `selection`,
`eq_source`, `draws_per_item`, `code_timeout`, `out_root` (per-draw outputs, e.g. on scratch), `keep_pickle`,
`ida_extra_keys`, `meta_data`, `n_same_sig`, `tau_e` (energy confinement time per draw from the I2 netCDF of bouquet_prep:
`{"path": ..., "variable": ...}`, see `parallel.tau_e_lookup`), `spot_check` (`true` or `SpotCheck` fields, e.g. `{"fraction": 0.01, "scans": [...], "scan_vals": {"<scan>": [...]}}`: a seeded coin per
draw picks `fraction` of the ok draws, which then run `spot_check.spot_check_tests` (quick values, or `scan_vals`; the draw's settings) inside
the same item; a failed scan is recorded in the draw's `spot_errors`, never fails the draw; the store gets
`spot_check_flag(nn, code)`: -1 not checked, 0 passed, 1 flagged). At 1 % this adds about 20 % compute. Coupled runs set `chi_model: "total"` and `tau_e_label: "tau_E_th"` in `tps_kwargs` (the archive's χ_⊥e/i profiles and τ_E reach every draw; the default `chi_model` is `"tau_E"`). Production sets `require_rdcon` in `tps_kwargs`, so a draw whose RDCON fails stops at
once instead of running the other codes first. Workers refuse to run if the TPS/GPEC/jGPEC build differs from the one
`init` recorded (`--allow-build-drift`).

**Units of work.** A bouquet unit `<shot>_<t_ms>_<variant>` (one archive) is cut into items `<unit_id>.c<k>`
of `draws_per_item` draws; a worker claims an item and runs it in a child process. The draw is the restart unit.

**Layout.**
```
<study>/_queue/tps/            fsqueue state; study.json (settings); results/<item>.json
<out_root>/<shot>/tps/<unit_id>/draws/<count>.nc     combined_xr of one draw
<out_root>/<shot>/tps/<unit_id>/draws/<count>.json   status, written last: ok | error | timeout | crashed
<out_root>/<shot>/tps/<unit_id>/draws/<count>.spot/    spot-checks of a picked draw (spot_check)
<out_root>/<shot>/tps/<unit_id>/logs/<item>.log
<study>/<shot>/tps/<shot>.zarr, <shot>_inputs.nc, compile_state.json
```
Each run in a store has `case_key` (`<unit_id>_<count>`), `shot_id`, `t_ms`, `draw` and `variant`.

**Failures.**
- A draw that fails in the pipeline is a result (`error`, with its signature); its item still succeeds.
- A draw that times out (`code_timeout`) or whose process dies is retried once, then final (`timeout`/`crashed`).
- An item fails (and is retried by the queue, at most `MAX_FAILURES` times) for a first timeout or an error outside
  the draws. `ImportError`, `MemoryError`, a full or read-only disk, a build mismatch, or `n_same_sig` draws in a row
  failing with one signature across two units mark the worker broken: it exits. When workers of
  `BROKEN_JOBS_STOP` jobs break within an hour, the queue writes `STOP`.

**Leases and duplicates.** A worker holds a heartbeat lease on its item; a lease whose heartbeat is not seen to
change for `STALE_S` (300 s; `_queue/tps/config.json`) is taken over. A live worker whose filesystem stalls that
long can lose its lease, and the item then runs twice: harmless (each run writes its own files atomically, and a
finished item is never marked failed), but logged ("lease was taken over").

**Stores.** `compile` appends by position along ragged dims (NaN-padded), as `multi_compile_zarr` does; `open_study`
pads and concatenates the shot stores the same way.

**Cost (omega, one CPU, production settings: n = 1-4, RDCON + STRIDE + jGPEC Galerkin, DIII-D wall).** About
24 min per draw (RDCON ~7.5, STRIDE ~6.5, jGPEC Galerkin ~9 including ~1.5 min of JIT per item) and a peak of
~21 GB per worker, set by jGPEC Galerkin at n = 4. Adding jGPEC Riccati makes it ~2.2 h per draw. RDCON at n = 3 on a
draw with 36 rational surfaces peaked at 26 GB; with ≥ 25 surfaces the RDCON in GPEC `OFT_interface` dfd771ca crashes
writing `delta_gw.out` (fixed on GPEC `bugfix/rdcon-gal-format-width`). Size workers per node by memory.

**Recommendations (Perlmutter).** Multi-node `regular` (or `preempt`, which every draw survives) jobs with one
launcher per node, sized to the backlog; single-node jobs for trickles and tails. Workers per node:
min(physical cores, memory / peak RSS per draw). Bouquet archives and the final zarr on CFS; `out_root` and worker
dirs on `$PSCRATCH` and `/tmp`.
