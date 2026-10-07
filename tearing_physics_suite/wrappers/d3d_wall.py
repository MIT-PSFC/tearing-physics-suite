"""DIII-D vacuum vessel wall of Fortran VACUUM (ishape=8: d3dwall/d3dvesl in vacuum_vac.f), for jGPEC wall files."""
import numpy as np

# d3dwall data: centre, minor radius, elongation and Fourier coefficients
R0, Z0, A0, E0 = 1.64, 0.0, 0.883941, 1.403702
RWI = np.array([0.1000000e+01, 0.5526794e-01, -0.1738114e+00, 0.1850757e-01, 0.3714965e-01, -0.2882647e-01,
                -0.2357329e-02, 0.9548103e-02, -0.1214923e-01, -0.1853416e-02, 0.6837493e-02, -0.1711245e-02,
                0.2270762e-02, 0.3689963e-02, -0.3959393e-02, -0.1098017e-02, 0.3745465e-02, -0.2157904e-03,
                -0.3977743e-03, -0.2725623e-03, -0.1005857e-02, -0.4579016e-05, 0.2396789e-02, -0.7057043e-03,
                0.1158347e-02, 0.3552319e-03])
ZWI = np.array([0.1000000e+01, -0.3236632e-01, -0.1629422e+00, 0.6013983e-01, 0.1167756e-01, -0.2579542e-01,
                0.1626464e-01, -0.2085857e-02, -0.9098639e-02, 0.1022163e-01, -0.4388253e-02, -0.9367258e-02,
                0.8308497e-02, 0.4765150e-02, -0.4611675e-02, -0.1121423e-02, -0.2501100e-03, 0.4282634e-03,
                0.2669702e-02, -0.1073800e-02, -0.2191338e-02, 0.1328267e-02, 0.5050959e-03, -0.5758863e-03,
                0.9348883e-03, 0.7094351e-03])


def d3d_wall_points(npts):
    """(R, Z) [m] of npts points equally spaced in arc from the outboard midplane (arc0 = 0), as d3dvesl.

    Clockwise in (R, Z) (Z = -e0*a0*sum(...)), periodic with no repeated endpoint.
    """
    arc = 2*np.pi*np.arange(npts)/npts
    k = np.arange(1, len(RWI) + 1)
    r = R0 + A0*np.cos(np.outer(arc, k)) @ RWI
    z = Z0 - E0*A0*np.sin(np.outer(arc, k)) @ ZWI
    return r, z


def write_jgpec_wall_file(path, npts=480):
    """Write the DIII-D wall in jGPEC's wall-file format (npts, wcentr, header, rows 'i R Z'); returns path."""
    r, z = d3d_wall_points(npts)
    lines = [f'{npts}', f'{R0:.16e}', 'i  R  Z (DIII-D wall, VACUUM ishape=8)']
    lines += [f'{i + 1:6d} {ri:24.16e} {zi:24.16e}' for i, (ri, zi) in enumerate(zip(r, z))]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path
