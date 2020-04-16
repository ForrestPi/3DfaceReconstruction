import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
import cv2
import modules.mesh as mesh

def render(source_dir):
    img_list = []
    for _root, _dirs, _files in os.walk(source_dir):
        for _file in _files:
            path = os.path.join(_root, _file)
            des_dir = _root.replace('swap', '2dimage')
            if not os.path.exists(des_dir): 
                os.makedirs(des_dir)
            vertices, colors, triangles = mesh.interact.load_obj_with_colors(path)
            triangles = triangles - 1
            h = 480
            w = 640
            fitted_image = mesh.render.render_colors(vertices, triangles, colors, h, w) * 255.
            fitted_image = fitted_image.astype('uint8')
            out_path1 = path.replace("swap", "2dimage").replace('obj', 'jpg')
            out_path2 = path.replace("swap", "2dimageconcat").replace('obj', 'jpg')
            frame = int(path.split('/')[-1].split('_')[1])
            sourceB_path = path.replace('swap/003_000', 'frames/003').replace('obj', 'jpg')
            sourceB_path = sourceB_path.split('.')[0][:-2] + '.jpg'
            sourceB_img = cv2.imread(sourceB_path, 1)
            print(out_path1, sourceB_path, frame)
            img = np.concatenate([sourceB_img, fitted_image], 1)
            img_list.append([frame, fitted_image])
            cv2.imwrite(out_path1, fitted_image)
            cv2.imwrite(out_path2, img)
            

def write_video(img_list, path, fps=25, w=640, h=480):
    fourcc =  cv2.cv.CV_FOURCC(*'XVID')
    video = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for img in img_list:
        video.write(img)
    video.release()

source_dir = "/mnt/mfs/yiling/records/Deepfake/face2face/swap/003_000"
render(source_dir)
