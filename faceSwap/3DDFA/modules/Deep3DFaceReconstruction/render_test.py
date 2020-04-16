import sys
import os
import cv2
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import mesh

path = 'out/003_000/0000.obj'
vertices, colors, triangles = mesh.interact.load_obj_with_colors(path)
colors = colors[:, ::-1]
vertices *= 112
x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
z_min, z_max = min(vertices[:, 2]), max(vertices[:, 2])
print(x_min, y_min, z_min, x_max, y_max, z_max)
vertices[:, 0] -= x_min
vertices[:, 1] -= y_min
vertices[:, 2] -= z_min
h = 224
w = 224
fitted_image = mesh.render.render_colors(vertices, triangles, colors, h, w)
print(fitted_image.min(), fitted_image.max(), fitted_image.sum())
cv2.imwrite('test.jpg', fitted_image.astype('uint8'))
