"""
create by: yiling
create time: 2020.02.19
"""
import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

cdef extern from "deformation_core.h":
     void _deformation_core(float* verticesSA, float* verticesTA, int* triVertexIds, \
                            int nVertices, int nTriangles, float* verticesSBs, int nSBs, \
                            float* results)

def deformation_core(np.ndarray[float, ndim=2, mode='c'] verticesSA not None, \
                     np.ndarray[float, ndim=2, mode='c'] verticesTA not None, \
                     np.ndarray[int, ndim=2, mode='c'] triVertexIds not None, \
                     int nVertices, int nTriangles, \
                     np.ndarray[float, ndim=3, mode='c'] verticesSBs not None, \
                     int nSBs, \
                     np.ndarray[float, ndim=3, mode='c'] results not None):
    _deformation_core(<float*> np.PyArray_DATA(verticesSA), \
                      <float*> np.PyArray_DATA(verticesTA), \
                      <int *>  np.PyArray_DATA(triVertexIds), \
                      nVertices, nTriangles, \
                      <float*> np.PyArray_DATA(verticesSBs), \
                      nSBs, \
                      <float*> np.PyArray_DATA(results))
