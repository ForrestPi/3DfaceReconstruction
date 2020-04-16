import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))
import numpy as np

import modules.deformation as deformation
import modules.mesh as mesh

sourceA = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/000/000_f0001_0.obj"
sourceB = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/000/000_f0020_0.obj"
targetA = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/003/003_f0091_0.obj"

verticesSA, colorsSA, triangles = mesh.interact.load_obj_with_colors(sourceA)
verticesSB, colorsSB, triangles = mesh.interact.load_obj_with_colors(sourceB)
verticesTA, colorsTA, triangles = mesh.interact.load_obj_with_colors(targetA)
verticesSBs = verticesSB[np.newaxis, :]
verticesSA = verticesSA.astype(np.float32)
verticesTA = verticesTA.astype(np.float32)
triangles = triangles.astype(np.int32)
colorsSA = colorsSA.astype(np.float32)
verticesSBs = verticesSBs.astype(np.float32)
print(verticesSB.shape, type(verticesSBs))
nVertices = verticesSA.shape[0]
nTriangles = triangles.shape[0]
nSBs = 1
results = np.zeros((nSBs, nVertices, 3), dtype=np.float32)
deformation.deformation.deformation_core(verticesSA, verticesTA, triangles, nVertices,\
    nTriangles, verticesSBs, nSBs, results)
mesh.interact.write_obj_with_colors("out.obj", results[0], triangles, colorsTA)
