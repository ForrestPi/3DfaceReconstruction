import numpy as np
from .cython import mesh_core_cython

def render_colors(vertices, triangles, colors, h, w, c=3, BG=None):
    """
    Args:
        vertices: [n_vertices, 3]
        triangles: [n_vertices, 3]
        colors: [n_vertices, 3]
        h: height
        w: width
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image
    """
    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG.astype(np.float32)
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()

    mesh_core_cython.render_colors_core(
        image, vertices, triangles, colors, depth_buffer, \
        vertices.shape[0], triangles.shape[0], h, w, c)

    return image

