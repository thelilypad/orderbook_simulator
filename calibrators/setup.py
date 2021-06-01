from distutils.core import setup

import numpy as np
from Cython.Build import cythonize

setup(name="bates_calibrator",
      include_dirs=[np.get_include()],
      ext_modules=cythonize(module_list=["utils.pyx", "bates_mcmc_calibrator.pyx"],
                            compiler_directives={'language_level': "3"}))
