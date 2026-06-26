"""GPR profile perturbation routines
===================================

Gaussian-process-regression (GPR) based sampling of smooth 1-D MHD
profiles. Original scripts written by Daniel Burgess.

Provides:
  - ``GPRProfilePerturber`` – GPR-based profile perturbation class.
  - ``generate_perturbed_GPR`` – convenience one-call wrapper.
  - ``verify_gpr_statistics`` – Monte-Carlo validation of GPR sampling.
  - ``calc_cylindrical_li_proxy`` – cylindrical :math:`l_i` proxy from
    a 1-D :math:`j_\\phi` profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Optional

# ====================================================================
#  GPR profile perturber
# ====================================================================
class GPRProfilePerturber:
    r"""Gaussian-process perturber for smooth 1-D MHD profiles.

    Generates correlated random perturbations whose pointwise
    standard deviation is set **directly** by the user-supplied
    :math:`\sigma(x)` (experimental uncertainty in profile units).

    Internally the kernel amplitude is fixed to unity so that

    .. math::

        \operatorname{Cov}\!\bigl[\delta f(x),\,\delta f(x')\bigr]
        = \sigma(x)\;\sigma(x')\; k_1\!\bigl(x,x'\bigr)

    where :math:`k_1` is the unit-variance base kernel and the
    marginal standard deviation at every grid point equals the
    input uncertainty exactly:

    .. math::

        \sigma_{\rm GP}(x) = \sigma(x)

    Parameters
    ----------
    kernel_func : str
        Kernel name: ``'rbf'`` (:math:`C^\infty`) or
        ``'matern52'`` (:math:`C^2`).
    length_scale : float
        Correlation length in :math:`\hat\psi` units.
        Controls *wiggliness* of the draws but does **not** affect
        the pointwise amplitude.
    """

    _ALLOWED_KERNELS = {"rbf", "matern52"}

    def __init__(
        self,
        kernel_func: str = "rbf",
        length_scale=0.1,
    ):
        if kernel_func not in self._ALLOWED_KERNELS:
            raise ValueError(
                f"Kernel '{kernel_func}' not in {self._ALLOWED_KERNELS}.  "
                "Rougher kernels produce non-differentiable profiles "
                "unsuitable for MHD inputs."
            )
        self.kernel_func = kernel_func

        # length_scale: scalar or 1-D array (non-stationary Gibbs kernel)
        ls = np.asarray(length_scale, dtype=np.float64)
        self._ls_is_array = ls.ndim >= 1 and ls.size > 1
        self.length_scale = ls

        self._kernel = {
            "rbf": self._rbf_kernel,
            "matern52": self._matern52_kernel,
        }[self.kernel_func]

    # ---- unit-variance kernels --------------------------------------
    def _ell_matrices(self, n1: int, n2: int):
        r"""Build per-pair length-scale matrices for non-stationary kernels.

        Returns ``(ell_i, ell_j)`` each of shape ``(n1, n2)`` such that
        ``ell_i[a, b] = ell[a]`` and ``ell_j[a, b] = ell[b]``.
        For a scalar length scale, returns ``(scalar, scalar)`` unchanged.
        """
        if not self._ls_is_array:
            return self.length_scale, self.length_scale
        ell = self.length_scale
        return ell[:n1, None] * np.ones((1, n2)), np.ones((n1, 1)) * ell[:n2]

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r"""Squared-exponential (Gibbs) kernel with unit variance.

        For a scalar length scale this is the standard stationary RBF.
        For a spatially-varying :math:`\ell(x)` it uses the Gibbs
        non-stationary kernel:

        .. math::

            K(x_i, x_j) = \sqrt{\frac{2\,\ell_i\,\ell_j}
                                      {\ell_i^2 + \ell_j^2}}
                           \exp\!\Bigl(-\frac{d_{ij}^2}
                                             {\ell_i^2 + \ell_j^2}\Bigr)

        which preserves :math:`K(x, x) = 1`.
        """
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        ell_i, ell_j = self._ell_matrices(len(X1), len(X2))

        if not self._ls_is_array:
            return np.exp(-0.5 * (d / ell_i) ** 2)

        ell2_sum = ell_i**2 + ell_j**2
        prefactor = np.sqrt(2.0 * ell_i * ell_j / ell2_sum)
        return prefactor * np.exp(-d**2 / ell2_sum)

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r"""Matérn-5/2 (Gibbs) kernel with unit variance.

        Non-stationary extension: replaces the scalar :math:`\ell` with
        the geometric mean :math:`\bar{\ell} = \sqrt{\ell_i \ell_j}` and
        applies the same Gibbs prefactor as the RBF kernel.
        """
        d = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "euclidean")
        ell_i, ell_j = self._ell_matrices(len(X1), len(X2))

        if not self._ls_is_array:
            s = np.sqrt(5.0) * d / ell_i
            return (1.0 + s + s**2 / 3.0) * np.exp(-s)

        ell2_sum = ell_i**2 + ell_j**2
        prefactor = np.sqrt(2.0 * ell_i * ell_j / ell2_sum)
        ell_geom = np.sqrt(ell_i * ell_j)
        s = np.sqrt(5.0) * d / ell_geom
        return prefactor * (1.0 + s + s**2 / 3.0) * np.exp(-s)

    # ---- core sampling method ----------------------------------------
    def precompute_factor(
        self,
        psi_N: np.ndarray,
        sigma_profile: np.ndarray,
    ) -> None:
        r"""Pre-compute and cache GPR eigen-factor.

        After calling this, use :meth:`draw_from_factor` to generate
        samples without repeating the O(n³) eigendecomposition -> np.linalg.eigh

        Parameters
        ----------
        psi_N : ndarray
            1-D normalised flux grid.
        sigma_profile : ndarray
            1-D experimental uncertainty **in profile units** -- this
            becomes the GP's marginal standard deviation at every
            grid point.
        """
        # 1. Unit-variance base kernel
        K = self._kernel(psi_N, psi_N)          # K(x,x) = 1

        # 2. Scale by σ(x):  C_ij = σ_i · σ_j · K_ij
        #    ⟹  C(x,x) = σ(x)²  ⟹  marginal std = σ(x)   ✓
        S = np.outer(sigma_profile, sigma_profile)
        K_scaled = K * S

        # 3. Eigen-decomposition (symmetric → eigh)
        vals, vecs = np.linalg.eigh(K_scaled)
        vals = np.maximum(vals, 0.0)
        self._cached_vecs = vecs
        self._cached_sqrt_vals = np.sqrt(vals)

    def draw_from_factor(
        self,
        input_profile: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r"""Draw perturbed profiles whose pointwise :math:`1\sigma`
        matches the uncertainty supplied to :meth:`precompute_factor` exactly.

        Call :meth:`precompute_factor` first.

        Parameters
        ----------
        input_profile : ndarray
            1-D baseline profile (GP mean).
        n_samples : int
            Number of independent profile draws.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        ndarray, shape ``(n_samples, len(input_profile))``
        """
        n = len(input_profile)

        # 4. Sample:  δf = V diag(√λ) z,   z ~ N(0, I)
        z = rng.standard_normal((n, n_samples))
        perturbations = self._cached_vecs @ (self._cached_sqrt_vals[:, None] * z)

        # 5. Perturbed profiles  →  (n_samples, n_points)
        return input_profile[None, :] + perturbations.T

    def generate_profiles(
        self,
        psi_N: np.ndarray,
        input_profile: np.ndarray,
        sigma_profile: np.ndarray,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        r"""Draw perturbed profiles whose pointwise :math:`1\sigma`
        matches the supplied experimental uncertainty exactly.

        Parameters
        ----------
        psi_N : ndarray
            1-D normalised flux grid.
        input_profile : ndarray
            1-D baseline profile (GP mean).
        sigma_profile : ndarray
            1-D experimental uncertainty **in profile units** -- this
            becomes the GP's marginal standard deviation at every
            grid point.
        n_samples : int
            Number of independent draws.
        rng : numpy.random.Generator or None
            ``None`` creates a fresh unseeded generator.

        Returns
        -------
        ndarray, shape ``(n_samples, len(psi_N))``
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(psi_N)

        # 1. Unit-variance base kernel
        K = self._kernel(psi_N, psi_N)          # K(x,x) = 1

        # 2. Scale by σ(x):  C_ij = σ_i · σ_j · K_ij
        #    ⟹  C(x,x) = σ(x)²  ⟹  marginal std = σ(x)   ✓
        S = np.outer(sigma_profile, sigma_profile)
        K_scaled = K * S

        # 3. Eigen-decomposition (symmetric → eigh)
        vals, vecs = np.linalg.eigh(K_scaled)
        vals = np.maximum(vals, 0.0)

        # 4. Sample:  δf = V diag(√λ) z,   z ~ N(0, I)
        z = rng.standard_normal((n, n_samples))
        perturbations = vecs @ (np.sqrt(vals)[:, None] * z)

        # 5. Perturbed profiles  →  (n_samples, n_points)
        return input_profile[None, :] + perturbations.T

# ====================================================================
#  Spatially-varying length scale helpers
# ====================================================================
def sigmoid_length_scale(
    psi_N: np.ndarray,
    ls_core: float = 0.3,
    ls_edge: float = 0.1,
    psi_transition: float = 0.7,
    steepness: float = 20.0,
) -> np.ndarray:
    r"""Build a sigmoid length-scale profile over normalised flux.

    .. math::

        \ell(\hat{\psi}) = \ell_{\rm core}
            - \frac{\ell_{\rm core} - \ell_{\rm edge}}
                   {1 + \exp\!\bigl[-k\,(\hat{\psi} - \hat{\psi}_t)\bigr]}

    Parameters
    ----------
    psi_N : ndarray
        1-D normalised flux grid.
    ls_core : float
        Correlation length in the core (:math:`\hat{\psi} \ll \hat{\psi}_t`).
    ls_edge : float
        Correlation length at the edge (:math:`\hat{\psi} \gg \hat{\psi}_t`).
    psi_transition : float
        Centre of the sigmoid transition.
    steepness : float
        Steepness *k* of the sigmoid (larger = sharper transition).

    Returns
    -------
    ndarray
        1-D array of length scales, same shape as ``psi_N``.
    """
    return ls_core - (ls_core - ls_edge) / (
        1.0 + np.exp(-steepness * (psi_N - psi_transition))
    )

# ====================================================================
#  Convenience wrapper
# ====================================================================
def generate_perturbed_GPR(
    xdata: np.ndarray,
    profile: np.ndarray,
    sigma_profile: Optional[np.ndarray] = None,
    length_scale=0.25,
    n_samples: int = 1,
    kernel_func: str = "rbf",
    rng: Optional[np.random.Generator] = None,
    diag_plot: bool = False,
) -> np.ndarray:
    r"""One-call wrapper: perturb a 1-D profile with a GPR draw.

    The ``sigma_profile`` input is the experimental
    :math:`1\sigma` uncertainty **in the same units as the profile**.
    It maps directly to the GP marginal standard deviation -- no
    separate ``variance`` parameter is needed.

    Parameters
    ----------
    xdata : ndarray
        1-D normalised flux grid :math:`\hat{\psi}`.
    profile : ndarray
        1-D baseline profile (GP mean).
    sigma_profile : ndarray or None
        1-D experimental :math:`1\sigma` uncertainty **in profile
        units**.  ``None`` gives zero (no perturbation).
    length_scale : float or ndarray
        GPR correlation length (controls wiggliness).  A scalar gives
        a stationary kernel; a 1-D array (same length as *xdata*)
        gives a non-stationary Gibbs kernel with spatially-varying
        correlation length.  See :func:`sigmoid_length_scale`.
    n_samples : int
        Number of independent draws.
    kernel_func : str
        ``'rbf'`` or ``'matern52'``.
    rng : numpy.random.Generator or None
        Optional random generator.
    diag_plot : bool
        Show a three-panel diagnostic figure.

    Returns
    -------
    ndarray
        If ``n_samples == 1``: 1-D array of length ``len(xdata)``.
        Otherwise: 2-D array ``(n_samples, len(xdata))``.
    """
    if sigma_profile is None:
        sigma_profile = np.zeros_like(xdata)

    perturber = GPRProfilePerturber(
        kernel_func=kernel_func,
        length_scale=length_scale,
    )

    perturbed = perturber.generate_profiles(
        psi_N=xdata,
        input_profile=profile,
        sigma_profile=sigma_profile,
        n_samples=n_samples,
        rng=rng,
    )

    # ---- diagnostic figure ------------------------------------------
    if diag_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Panel 1: profile ± σ_exp  (exact 1σ band)
        ax1.plot(xdata, profile, "k-", lw=2, label="Original profile")
        ax1.fill_between(
            xdata,
            profile - sigma_profile,
            profile + sigma_profile,
            alpha=0.3,
            label=r"$\pm\,1\sigma_{\rm exp}$ envelope",
        )
        ax1.set_xlabel(r"$\hat{\psi}$")
        ax1.set_ylabel("Profile value")
        ax1.set_title(r"Input profile with experimental $1\sigma$ uncertainty")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: sigma profile
        ax2.plot(xdata, sigma_profile, "r-", lw=2)
        ax2.set_xlabel(r"$\hat{\psi}$")
        ax2.set_ylabel(r"$\sigma_{\rm exp}(x)$ [profile units]")
        ax2.set_title("Experimental uncertainty profile")
        ax2.grid(True, alpha=0.3)

        # Panel 3: perturbed draws with band
        ax3.plot(xdata, profile, "k-", lw=3, label="Original")
        ax3.fill_between(
            xdata,
            profile - sigma_profile,
            profile + sigma_profile,
            alpha=0.15, color="gray",
            label=r"$\pm\,1\sigma_{\rm exp}$",
        )
        if n_samples == 1:
            ax3.plot(xdata, perturbed[0], "--", alpha=0.7, label="Perturbed")
        else:
            for i in range(perturbed.shape[0]):
                ax3.plot(
                    xdata, perturbed[i], "--", alpha=0.7,
                    label=f"Perturbed {i + 1}" if i < 10 else None,
                )
        ax3.set_xlabel(r"$\hat{\psi}$")
        ax3.set_ylabel("Profile value")
        ax3.set_title("Original and perturbed profiles")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    if n_samples == 1:
        return perturbed[0]
    return perturbed
