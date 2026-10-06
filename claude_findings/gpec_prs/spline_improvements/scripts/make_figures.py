"""Figures for the spline_improvements report.

usage: python make_figures.py RESULTS_ROOT FIGDIR
RESULTS_ROOT holds truth/, a4*/, a6/, a6_exact/ from the run_* scripts.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from gpec_profiles import sq_profiles, ldp_grid, gfile_profiles
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

ROOT, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
INK, MUTED = '#0b0b0b', '#52514e'
COL = {'values': '#2a78d6', 'hermite': '#eb6834', 'integrate': '#1baf7a', 'ifile': '#eda100'}
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'font.size': 10, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})

# 1. F' and F'' as GPEC's sq holds them, against TokaMaker truth (g257 file, mpsi = 512)
t = np.load(os.path.join(ROOT, 'truth/truth_profiles.npz'))
pad = json.load(open(os.path.join(ROOT, 'truth/truth_meta.json')))['psi_pad']
g = read_eqdsk(os.path.join(ROOT, 'truth/g257.geqdsk'))
x, F, p, ffp, pp = gfile_profiles(g)
psio = g['psibry'] - g['psimag']
xs_sq = ldp_grid(1e-4, 0.985 / (1 - pad), 512)
xx = np.linspace(0.40, 0.64, 3000)
Fp_t = np.interp(xx * (1 - pad), t['psi_N'], t['Fp']) * psio * (1 - pad)
Fpp_t = np.interp(xx * (1 - pad), t['psi_N'], np.gradient(t['Fp'], t['psi_N'])) * psio * (1 - pad) ** 2
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for m in ('values', 'hermite', 'integrate'):
    fs, _ = sq_profiles(x, F, p, ffp, pp, m, xs_sq)
    ax[0].plot(xx, fs(xx, 1), color=COL[m], label=m)
    ax[1].plot(xx, fs(xx, 2), color=COL[m], label=m, lw=1.5 if m != 'integrate' else 2.2)
for a, y, lab in ((ax[0], Fp_t, r"$dF/d\psi_N$  [T m]"), (ax[1], Fpp_t, r"$d^2F/d\psi_N^2$  [T m]")):
    a.plot(xx, y, color=INK, ls='--', lw=1.5, label='TokaMaker truth')
    a.set_xlabel(r'$\psi_N$'); a.set_ylabel(lab)
ax[1].legend(loc='upper right', ncol=2)
fig.suptitle("F as GPEC's sq holds it, from a TokaMaker g-file (257 nodes, single-precision values), mpsi = 512",
             color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'profile_source_Fpp.png'), dpi=150); plt.close(fig)

# 2. Delta'(2/1) vs mpsi
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
mp = (128, 256, 512)
res = {m: json.load(open(os.path.join(ROOT, d, 'profile_source_results.json')))
       for m, d in zip(mp, ('a4', 'a4_mpsi256', 'a4_mpsi512'))}
for i, (meth, mk) in enumerate((('values', 'o'), ('integrate', 's'))):
    for n, ls in zip((129, 257, 513), (':', '--', '-')):
        ax[i].plot(mp, [res[m][f'g{n}'][meth]['dp_diag'][0] for m in mp], ls=ls, marker=mk, ms=8,
                   color=COL[meth], label=f'g-file {n}x{n}')
    ax[i].set_title(f'g-file, profile_source = {meth}', color=INK, fontsize=10)
ex = {m: json.load(open(os.path.join(ROOT, 'a6_exact', f'exact_results_m{m}.json'))) for m in mp}
for name, ls in (('i129x257_exact', '--'), ('i257x513_exact', '-')):
    for a in ax:
        a.plot(mp, [ex[m][name]['dp_diag'][0] for m in mp], ls=ls, marker='D', ms=8, color=COL['ifile'],
               label=name.replace('_exact', ' i-file (exact crossings)'))
for a in ax:
    a.set_xscale('log', base=2); a.set_xticks(mp, [str(m) for m in mp]); a.set_xlabel('GPEC mpsi')
ax[0].set_ylabel(r"STRIDE $\Delta'$ (2/1)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'delta_prime_vs_mpsi.png'), dpi=150); plt.close(fig)

# 3. GSE vs file size
a6 = json.load(open(os.path.join(ROOT, 'a6/ifile_results.json')))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
groups = {'g-file (efit, integrate)': ([v for k, v in a6.items() if k.startswith('g')], COL['integrate'], 's'),
          'OFT i-file as written': ([v for k, v in a6.items() if k.startswith('i') and not k.endswith('_ffp')], COL['hermite'], 'o')}
ex128 = ex[128]
groups['i-file, exact psi crossings'] = (list(ex128.values()), COL['ifile'], 'D')
for lab, (vals, c, mk) in groups.items():
    ax.scatter([v['bytes'] / 1e6 for v in vals], [v['int_med'] for v in vals], color=c, marker=mk, s=40, label=lab,
               edgecolor='white', linewidth=1)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('file size [MB]'); ax.set_ylabel('GPEC GSE, surface-integrated, median')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'gse_vs_filesize.png'), dpi=150); plt.close(fig)
print('wrote', FIG)
