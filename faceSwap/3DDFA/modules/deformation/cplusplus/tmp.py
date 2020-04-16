import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
import cv2
import mesh

with open('kpt_ind.txt', 'r') as f:
    kpts = f.readlines()
kpts = [int(item.strip()) for item in kpts]

obj_file = 'out.obj'
obj_file = '/mnt/mfs/yiling/records/Deepfake/face2face/3dmm/000/000_0025_0.obj'
vertices, colors, triangles = mesh.interact.load_obj_with_colors(obj_file)
triangles = triangles - 1

kpts = vertices[kpts].tolist()
#h = w = 256
h = 480
w = 640
fitted_image = mesh.render.render_colors(vertices, triangles, colors, h, w) * 255.
print(triangles)
for p in kpts:
    x = round(p[0])
    y = round(p[1])
    cv2.circle(fitted_image, (x, y), 1, (0, 0, 255), thickness=1)
cv2.imwrite('out.jpg', fitted_image.astype('uint8'))
print(vertices.shape, colors.shape, triangles.shape)
