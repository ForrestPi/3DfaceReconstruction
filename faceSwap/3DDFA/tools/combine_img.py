import os
import cv2
import numpy as np

s = "000_003"
dir1 = "/mnt/mfs/yiling/records/Deepfake/face2face/swapdirect/%s" % s
dir2 = "/mnt/mfs/yiling/records/Deepfake/face2face/swap/%s" % s
des_dir = "/mnt/mfs/yiling/records/Deepfake/face2face/swap/combine/%s" % s
if not os.path.exists(des_dir):
    os.makedirs(des_dir)

dict1 = {}
for _root, _dirs, _files in os.walk(dir1):
    for _file in _files:
        path = os.path.join(_root, _file)
        if not path.endswith('.jpg'):
            continue
        img_name = path.split('/')[-1]
        dict1[img_name] = path

dict2 = {}
for _root, _dirs, _files in os.walk(dir2):
    for _file in _files:
        path = os.path.join(_root, _file)
        if not path.endswith('.jpg'):
            continue
        img_name = path.split('/')[-1]
        dict2[img_name] = path

for key in dict1:
    img1 = cv2.imread(dict1[key])
    img2 = cv2.imread(dict2[key])
    BG = img1[:, 0: 640 * 2, :]
    swap1 = img1[:, 640 * 2: , :]
    swap2 = img2[:, 640 * 2: , :]
    swap = np.concatenate([swap1, swap2], 1)
    img = np.concatenate([BG, swap], 0)
    path = os.path.join(des_dir, key)
    print(path)
    cv2.imwrite(path, img)
