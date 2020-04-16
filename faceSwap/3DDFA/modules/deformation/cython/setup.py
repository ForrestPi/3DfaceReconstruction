
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    name='deformation',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("deformation", \
              sources=["deformation.pyx", "deformation_core.cpp", "DeformationTransfer.cpp"], \
              language='c++', include_dirs=[numpy.get_include()])]
)
