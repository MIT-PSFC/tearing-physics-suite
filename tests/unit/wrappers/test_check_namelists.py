import os
from pathlib import Path

import pytest

from tearing_physics_suite.wrappers.check_namelists import (
    check_namelists,
    compare,
    parse_fortran_namelists,
    parse_namelist_file,
    parse_pest3_flags,
    pest3_command_flags,
)

GPEC = Path(os.environ['TPSHOME']) / 'submodules' / 'GPEC'


def test_fixed_form_namelists(tmp_path):
    f = tmp_path / 'a.f'
    f.write_text("c     comment line\n"
                 "      NAMELIST/ctl/bal_flag,MAT_flag, ! trailing comment\n"
                 "     $     nn,qlow\n"
                 "      namelist /out/ x /out2/ y, z\n")
    assert parse_fortran_namelists([f]) == {'ctl': {'bal_flag', 'mat_flag', 'nn', 'qlow'}, 'out': {'x'},
                                            'out2': {'y', 'z'}}


def test_free_form_namelists(tmp_path):
    f = tmp_path / 'a.f90'
    f.write_text("namelist / grp / a, b, &\n   c   ! c is last\n")
    assert parse_fortran_namelists([f]) == {'grp': {'a', 'b', 'c'}}


def test_written_namelist_parse(tmp_path):
    f = tmp_path / 'x.in'
    f.write_text("&MODES\n   mth = 480\n   xiin(1:9) = 0 1\n/\n&UA_DIAGNOSE_LIST\n    uad%flag=f\n/\n")
    assert parse_namelist_file(f) == {'modes': {'mth', 'xiin'}, 'ua_diagnose_list': {'uad'}}


def test_pest3_flags(tmp_path):
    hh = tmp_path / 'pest3.hh'
    hh.write_text("switch(c){\n case 'n':\n case 'b': break;\n/*    case 'e':\n*/\n}")
    assert parse_pest3_flags(hh) == {'n', 'b'}
    assert pest3_command_flags('./pest3x -iefit -n1 -k"100 -70" -x0') == {'i', 'n', 'k', 'x'}


def test_compare_classifies():
    (r,) = compare('c', 'f.in', {'g': {'a', 'b', 'z'}}, {'g': {'a', 'm'}, 'h': {'b'}})
    assert r.both == ['a'] and r.missing_in_tps == ['m'] and r.unknown_to_code == ['z']
    assert r.wrong_group == ['b (in h)'] and not r.ok
    (u,) = compare('c', 'f.in', {'x': {'a'}}, {'g': {'a'}})
    assert u.unread_group and u.ok


needs_sources = pytest.mark.skipif(not (GPEC / 'rdcon').is_dir(), reason='GPEC sources not present')


@needs_sources
def test_tps_inputs_match_code_sources():
    reports, _ = check_namelists()
    assert [f'{r.file}/{r.group}: {r.unknown_to_code + r.wrong_group}' for r in reports if not r.ok] == []


@needs_sources
@pytest.mark.xfail(strict=True, reason='dump_MRE_data is written to &RDCON_CONTROL but GPEC OFT_interface rdcon does not declare it')
def test_tps_optional_inputs_match_code_sources():
    reports, _ = check_namelists(all_paths=True)
    assert all(r.ok for r in reports)
