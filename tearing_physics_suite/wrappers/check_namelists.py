"""Compare the inputs TPS writes with the inputs each code's source declares.

For every input file (and PEST3's command-line flags) it reports:
  missing in TPS   - declared by the code, never set by TPS (the code default applies);
  UNKNOWN to code  - set by TPS, not declared by the code (error: the namelist read fails);
  wrong group      - set by TPS in a namelist group the code does not declare it in (error);
  note             - a group TPS writes that the code never declares (skipped when read).
Nothing is compiled or run: the Fortran/C sources are parsed, and the TPS writers write into a
temporary directory.

    python -m tearing_physics_suite.wrappers.check_namelists [--codes rdcon stride pest3] [--all-paths]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from tearing_physics_suite.utils import tps_home

EQ_NAME = 'g147131.02300_DIIID_KEFIT'
# input file -> (code, source files declaring its namelists, relative to the GPEC tree)
GPEC_FILES = {
    'rdcon.in': ('rdcon', ['rdcon/*.f', 'rdcon/*.F']),
    'stride.in': ('stride', ['stride/*.f', 'stride/*.F']),
    'equil.in': ('equil', ['equil/equil.f']),
    'vac.in': ('vacuum', ['vacuum/vacuum_io.f']),
}
CODE_FILES = {'rdcon': ('rdcon.in', 'equil.in', 'vac.in'), 'stride': ('stride.in', 'equil.in', 'vac.in')}
# Extra write configurations that switch on optional branches (--all-paths)
ALL_PATH_CONFIGS = [dict(vac_flag='f', ode_flag='f'), dict(dump_MRE_data='t'), dict(eq_type="'ldp_i'"),
                    dict(Zeff={'x': [0.0, 0.5, 1.0], 'y': [1.5, 2.0, 2.5]}), dict(gal_xmin_flag='t')]


@dataclass
class GroupReport:
    """Comparison for one input file and namelist group (PEST3: group 'flags')."""
    code: str
    file: str
    group: str
    both: list = field(default_factory=list)
    missing_in_tps: list = field(default_factory=list)
    unknown_to_code: list = field(default_factory=list)
    wrong_group: list = field(default_factory=list)
    unread_group: bool = False  # group written by TPS but declared nowhere in the code (skipped when read)

    @property
    def ok(self):
        return not (self.unknown_to_code or self.wrong_group)


# --------------------------------------------------------------------------------------------------
# Code side: parse the sources
# --------------------------------------------------------------------------------------------------
def _logical_lines(text, fixed_form):
    """Fortran source as logical lines: comments stripped, continuations joined."""
    out = []
    for raw in text.splitlines():
        if fixed_form and raw[:1] in ('c', 'C', '*', '!'):
            continue
        line = raw.split('!', 1)[0].rstrip()
        if fixed_form and len(line) > 5 and line[5] not in (' ', '0') and line[:5].strip() == '' and out:
            out[-1] += ' ' + line[6:]
        elif out and out[-1].endswith('&'):
            out[-1] = out[-1][:-1] + ' ' + line.lstrip().lstrip('&')
        else:
            out.append(line)
    return out


def parse_fortran_namelists(paths):
    """{group: set(names)} (lower case) from the NAMELIST statements in the given source files."""
    groups = {}
    for p in paths:
        text = Path(p).read_text(errors='replace')
        fixed = Path(p).suffix in ('.f', '.F')
        for line in _logical_lines(text, fixed):
            m = re.match(r'\s*namelist\b(.*)', line, re.I)
            if not m:
                continue
            for grp, names in re.findall(r'/\s*(\w+)\s*/([^/]*)', m.group(1)):  # /g1/ a,b /g2/ c
                groups.setdefault(grp.lower(), set()).update(n.strip().lower() for n in names.split(',') if n.strip())
    return groups


def parse_pest3_flags(hh_path):
    """Command-line flags (single letters) PEST3 accepts: the `case 'X':` entries of pest3.hh."""
    text = Path(hh_path).read_text(errors='replace')
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)  # drop commented-out cases
    return set(re.findall(r"case\s+'(\w)'\s*:", text))


def _git_describe(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'describe', '--tags', '--always', '--dirty'],
                              capture_output=True, text=True, timeout=20).stdout.strip() or 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


# --------------------------------------------------------------------------------------------------
# TPS side: what the writers write
# --------------------------------------------------------------------------------------------------
def parse_namelist_file(path):
    """{group: set(names)} (lower case) from a written namelist file (`&group ... /`, `name=value`)."""
    groups, grp = {}, None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith('&'):
            grp = s[1:].strip().lower()
            groups.setdefault(grp, set())
        elif s == '/':
            grp = None
        elif grp:
            m = re.match(r'([A-Za-z_]\w*)\s*(%|\(|=)', s)
            if m:
                groups[grp].add(m.group(1).lower())
    return groups


def tps_gpec_inputs(configs=({},)):
    """{file: {group: set(names)}} written by write_rdcon_stride_inputs, unioned over configs."""
    from tearing_physics_suite.wrappers.gpec_inputs import write_rdcon_stride_inputs
    out = {}
    for cfg in configs:
        with tempfile.TemporaryDirectory() as d:
            write_rdcon_stride_inputs(d, EQ_NAME, verbose=False, **cfg)
            for f in GPEC_FILES:
                if os.path.exists(os.path.join(d, f)):
                    for g, names in parse_namelist_file(os.path.join(d, f)).items():
                        out.setdefault(f, {}).setdefault(g, set()).update(names)
    return out


class _Captured(Exception):
    pass


def tps_pest3_command(**pest3_kwargs):
    """The pest3x command line PEST3_resistive_calculation would run (captured, not run)."""
    import tearing_physics_suite.wrappers.pest3 as pest3
    cmd = {}

    def fake_system(c):
        cmd['c'] = c
        raise _Captured
    with tempfile.TemporaryDirectory() as d:
        Path(d, 'pest3x').write_text('')
        eq = Path(d, EQ_NAME)
        eq.write_text('')
        old, cwd = pest3.os.system, os.getcwd()
        pest3.os.system = fake_system
        try:
            pest3.PEST3_resistive_calculation(str(eq), 1, working_dir=os.path.join(d, 'work'), pest3_dir=d,
                                              **pest3_kwargs)
        except _Captured:
            pass
        finally:
            pest3.os.system = old
            os.chdir(cwd)
    return cmd.get('c', '')


def pest3_command_flags(cmd):
    """Single-letter flags in a pest3x command line (quoted values skipped)."""
    cmd = re.sub(r'"[^"]*"', '""', cmd)
    return set(re.findall(r'(?:^|\s)-(\w)', cmd.split('pest3x', 1)[-1]))


# --------------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------------
def compare(code, file, tps_groups, src_groups):
    """GroupReports for one written file: TPS {group: names} against source {group: names}."""
    reports = []
    for g in sorted(set(tps_groups) | set(src_groups)):
        if g not in tps_groups:
            continue  # declared by the code but not written: whole group at defaults
        tps, src = tps_groups[g], src_groups.get(g, set())
        if g not in src_groups:
            reports.append(GroupReport(code, file, g, unread_group=True))
            continue
        r = GroupReport(code, file, g, both=sorted(tps & src), missing_in_tps=sorted(src - tps))
        for n in sorted(tps - src):
            elsewhere = [h for h, names in src_groups.items() if n in names]
            (r.wrong_group if elsewhere else r.unknown_to_code).append(n + (f' (in {",".join(elsewhere)})' if elsewhere else ''))
        reports.append(r)
    return reports


def check_namelists(codes=('rdcon', 'stride', 'pest3'), gpec_dir=None, pest3_dir=None, all_paths=False):
    """Run the comparison. Returns (reports, versions)."""
    home = Path(tps_home())
    gpec_dir = Path(gpec_dir or home / 'submodules/GPEC')
    pest3_dir = Path(pest3_dir or home / 'submodules/PEST3')
    reports, versions = [], {}
    gpec_files = sorted({f for c in codes if c in CODE_FILES for f in CODE_FILES[c]})
    if gpec_files:
        versions['GPEC'] = _git_describe(gpec_dir)
        written = tps_gpec_inputs([{}] + (ALL_PATH_CONFIGS if all_paths else []))
        for f in gpec_files:
            owner, globs = GPEC_FILES[f]
            srcs = [p for g in globs for p in sorted(gpec_dir.glob(g))]
            if not srcs:
                raise FileNotFoundError(f'no sources for {f} under {gpec_dir}')
            reports += compare(owner, f, written.get(f, {}), parse_fortran_namelists(srcs))
    if 'pest3' in codes:
        versions['PEST3'] = _git_describe(pest3_dir)
        hh = pest3_dir / 'pest3' / 'pest3.hh'
        flags = pest3_command_flags(tps_pest3_command())
        reports += compare('pest3', 'pest3x command', {'flags': flags}, {'flags': parse_pest3_flags(hh)})
    return reports, versions


def format_reports(reports, versions, show_missing=True):
    lines = ['Sources: ' + ', '.join(f'{k} {v}' for k, v in versions.items())]
    for r in reports:
        if r.unread_group:
            lines.append(f'[note]  {r.code:7s} {r.file:15s} {r.group:18s} group not declared by the code (not read)')
            continue
        status = 'ok' if r.ok else 'ERROR'
        lines.append(f'[{status:5s}] {r.code:7s} {r.file:15s} {r.group:18s} {len(r.both)} matched')
        for n in r.unknown_to_code:
            lines.append(f'    UNKNOWN to code: {n}')
        for n in r.wrong_group:
            lines.append(f'    wrong group:     {n}')
        if show_missing and r.missing_in_tps:
            lines.append('    missing in TPS:  ' + ', '.join(r.missing_in_tps))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--codes', nargs='*', default=['rdcon', 'stride', 'pest3'])
    ap.add_argument('--gpec-dir')
    ap.add_argument('--pest3-dir')
    ap.add_argument('--all-paths', action='store_true', help='also write optional branches')
    ap.add_argument('--no-missing', action='store_true', help='hide names TPS leaves at code defaults')
    a = ap.parse_args(argv)
    reports, versions = check_namelists(a.codes, a.gpec_dir, a.pest3_dir, a.all_paths)
    print(format_reports(reports, versions, not a.no_missing))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == '__main__':
    sys.exit(main())
