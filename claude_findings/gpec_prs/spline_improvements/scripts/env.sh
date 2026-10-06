# Environment for the spline_improvements scripts (bash: source env.sh)
source /global/cfs/cdirs/m3195/stubenj9/tmdb/scripts/eq_stab/D3D_V1/FUSE_IDAlite_BOUQUET/omega/tmdb_env.sh >/dev/null 2>&1
source /global/cfs/cdirs/m3195/stubenj9/tmdb/src/tearing-physics-suite/tearing_physics_suite_env.sh >/dev/null 2>&1
export TPS=/global/cfs/cdirs/m3195/stubenj9/tmdb/src/tearing-physics-suite
export GPEC=$TPS/submodules/GPEC
export WT=$PSCRATCH/tmdb/build/gpec_wt
export UV_CACHE_DIR=$PSCRATCH/uv-cache UV_LINK_MODE=copy HDF5_USE_FILE_LOCKING=FALSE
