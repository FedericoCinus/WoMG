from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import platform
#import numpy as np
from collections import defaultdict

ext_args = defaultdict(list)

if platform.system() == "Darwin":
        ext_args['extra_compile_args'].append("-stdlib=libc++")
        
ext_modules = [Extension("n2i.graph", ['n2i/graph.pyx'], language="c++", **ext_args)]

setup(
    name = 'n2i graph',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
#    include_dirs = [np.get_include()]
)

