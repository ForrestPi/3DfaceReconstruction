import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

cdef extern from "mesh_core.h":
    void _render_colors_core(float* image, float* vertices, int* triangles, float* colors, float* depth_buffer, \
                             int n_vertices, int n_triangles, int h, int w, int c)

def render_colors_core(np.ndarray[float, ndim=3, mode='c'] image not None, \
                       np.ndarray[float, ndim=2, mode='c'] vertices not None, \
                       np.ndarray[int, ndim=2, mode='c'] triangles not None, \
                       np.ndarray[float, ndim=2, mode='c'] colors not None, \
                       np.ndarray[float, ndim=2, mode='c'] depth_buffer not None, \
                       int n_vertices, int n_triangles, int h, int w, int c):
    _render_colors_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles), \
        <float*> np.PyArray_DATA(colors), <float*> np.PyArray_DATA(depth_buffer), \
        n_vertices, n_triangles, h, w, c
    )
