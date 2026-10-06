"""i-file (OFT save_ifile / GPEC eq_type='ldp_i') helpers.

read_ifile, write_ifile: sequential unformatted records, 4-byte markers, real*8.

append_derivatives: write a copy of an i-file with trailing FF' and p' records (psi in Wb/rad,
    as GPEC's read_eq_ldp_i reads them), taken from TokaMaker truth profiles.
gs_residual: Grad-Shafranov residual of the i-file's own R,Z(psi,theta), F, p, independent of GPEC.
"""
import struct
import numpy as np
from scipy.interpolate import CubicSpline

MU0 = 4e-7 * np.pi


def read_ifile(path):
    """Records of a sequential unformatted i-file (4-byte markers): dict with npsi, ntheta, psi, f, p, q, R, Z (npsi, ntheta) [, ffp, pp]."""
    raw = open(path, 'rb').read()
    recs, off = [], 0
    while off < len(raw):
        n = struct.unpack_from('i', raw, off)[0]
        recs.append(raw[off + 4:off + 4 + n])
        off += n + 8
    npsi, ntheta = struct.unpack('2i', recs[0])
    arr = [np.frombuffer(r, dtype='<f8') for r in recs[1:]]
    out = dict(npsi=npsi, ntheta=ntheta, psi=arr[0], f=arr[1], p=arr[2], q=arr[3],
               R=arr[4].reshape(npsi, ntheta), Z=arr[5].reshape(npsi, ntheta))
    if len(arr) >= 8:
        out['ffp'], out['pp'] = arr[6], arr[7]
    return out


def write_ifile(path, d):
    """Write an i-file (double precision) from a read_ifile dict, including ffp/pp records if present."""
    def rec(b):
        return struct.pack('i', len(b)) + b + struct.pack('i', len(b))
    arrs = [d['psi'], d['f'], d['p'], d['q'], d['R'], d['Z']] + ([d['ffp'], d['pp']] if 'ffp' in d else [])
    with open(path, 'wb') as fh:
        fh.write(rec(struct.pack('2i', d['npsi'], d['ntheta'])))
        for a in arrs:
            fh.write(rec(np.ascontiguousarray(a, dtype='<f8').tobytes()))


def append_derivatives(src, dst, truth, psi_bounds):
    """Copy src to dst and append FF'(psi) and p'(psi) records evaluated from the truth npz on the file's psi grid."""
    d = read_ifile(src)
    psi_axis, psi_edge = psi_bounds[1], psi_bounds[0]
    x = (psi_axis - d['psi']) / (psi_axis - psi_edge)
    ffp = CubicSpline(truth['psi_N'], truth['FFp'])(x)
    pp = CubicSpline(truth['psi_N'], truth['pp'])(x)
    with open(dst, 'wb') as fh:
        fh.write(open(src, 'rb').read())
        for a in (ffp, pp):
            b = np.ascontiguousarray(a, dtype='<f8').tobytes()
            fh.write(struct.pack('i', len(b)) + b + struct.pack('i', len(b)))


def gs_residual(d, ffp, pp):
    """Relative GS residual |Delta* psi + mu0 R^2 p' + FF'| / (|mu0 R^2 p'| + |FF'|) on the i-file grid.

    psi in Wb/rad; theta derivatives spectral (periodic, duplicate endpoint dropped), psi derivatives
    from not-a-knot cubic splines. Delta* psi = R^2/J [d_psi(J g^pp/R^2) + d_theta(J g^pt/R^2)], J = R D.
    """
    psi = d['psi']
    R, Z = d['R'][:, :-1], d['Z'][:, :-1]
    nt = R.shape[1]
    k = np.fft.fftfreq(nt, 1.0 / nt) * 2 * np.pi        # theta in [0, 1)
    dth = lambda a: np.real(np.fft.ifft(1j * k * np.fft.fft(a, axis=1), axis=1))
    sg = np.sign(psi[-1] - psi[0])                       # splines need increasing abscissae
    dps = lambda a: sg * CubicSpline(sg * psi[1:], a[1:], axis=0)(sg * psi, 1)  # skip the axis row
    Rp, Zp, Rt, Zt = dps(R), dps(Z), dth(R), dth(Z)
    D = Rp * Zt - Rt * Zp
    J = R * D
    gpp = (Rt ** 2 + Zt ** 2) / D ** 2
    gpt = -(Rp * Rt + Zp * Zt) / D ** 2
    lap = R ** 2 / J * (dps(J * gpp / R ** 2) + dth(J * gpt / R ** 2))
    src = MU0 * R ** 2 * pp[:, None] + ffp[:, None]
    return np.abs(lap + src) / (np.abs(MU0 * R ** 2 * pp[:, None]) + np.abs(ffp[:, None]))
