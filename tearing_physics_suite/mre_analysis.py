# Python functions to call GPEC and PEST3 fortran codes for delta prime calculations

import os
import sys
import shutil
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import CubicSpline

from tearing_physics_suite.environment import home_dir