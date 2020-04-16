import numpy as np
import os

from .cython import mesh_core_cython

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    """ Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: (n_vertices, 3)
        triangles: (n_vertices, 3)
        colors: (n_vertices, 3)
    """
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)

def load_obj_with_colors(obj_name):
    vertices = []
    colors = []
    triangles = []
    with open(obj_name, 'r') as f:
        for line in f:
            tmp = line.strip().split()
            if tmp[0] == 'v':
                vertices.append([tmp[1], tmp[2], tmp[3]])
                colors.append([tmp[4], tmp[5], tmp[6]])
            elif tmp[0] == 'f':
                triangles.append([int(tmp[3]) - 1, int(tmp[2]) - 1, int(tmp[1]) - 1])
    vertices = np.array(vertices, dtype='float')
    colors = np.array(colors, dtype='float')
    triangles = np.array(triangles, dtype='int')
    return vertices, colors, triangles

