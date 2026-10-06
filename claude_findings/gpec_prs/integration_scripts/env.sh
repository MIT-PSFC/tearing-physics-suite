# Environment for the integration scripts (bash: source env.sh). Machine paths come from eq_stab's tmdb.env.
export TPS=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
source "$TPS/../../scripts/eq_stab/D3D_V1/FUSE_IDAlite_BOUQUET/omega/tmdb_env.sh" >/dev/null 2>&1
source "$TPS/tearing_physics_suite_env.sh" >/dev/null 2>&1
export GPEC=$TPS/submodules/GPEC
export WT=${TMDB_BUILD:-$TPS/build}/gpec_wt
