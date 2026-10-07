"""Small xarray helpers shared by the physics modules."""
import numpy as np
import xarray as xr


def like(values, template):
    """values (array or scalar) on template's dims and coords, NaN where template is NaN.

    Replaces the `values + 0.0*template` idiom with the same result.
    """
    if isinstance(values, xr.DataArray):  # aligns by dim name, as before
        return values + 0.0 * template
    data = np.broadcast_to(np.asarray(values, dtype=np.float64) + 0.0, template.shape).copy()
    return xr.DataArray(data, dims=template.dims, coords=template.coords).where(template.notnull())


def interp_to_surfaces(ds, names, method='cubic'):
    """{name: ds[name] (on psi_n) interpolated onto the rational surfaces, on psi_n_rational's dims}."""
    psi = ds['psi_n_rational']
    out = ds[list(names)].interp(psi_n=psi.values, method=method)
    return {n: like(out[n].values, psi) for n in names}
