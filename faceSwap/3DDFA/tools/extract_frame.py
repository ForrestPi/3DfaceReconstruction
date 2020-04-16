import cv2
import os

def extract_frame_from_dir(source_root, des_root):
    for _root, _dirs, _files in os.walk(source_root):
        for _file in _files:
            path = os.path.join(_root, _file)
            video_name = path.split('/')[-1].split('.')[0]
            des_dir = os.path.join(des_root, video_name)
            if not os.path.exists(des_dir):
                os.makedirs(des_dir)
            cap = cv2.VideoCapture(path)
            num_frame = cap.get(7)
            print(path, num_frame)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                img_name = video_name + '_' + "%04d" % i
                img_path = os.path.join(des_dir, img_name + '.jpg')
                cv2.imwrite(img_path, frame)
                i += 1
                if not ret:
                    break
            print(i)
            cap.release()

source_root = "/mnt/mfs/yiling/records/Deepfake/face2face/origin_videos"
des_root = "/mnt/mfs/yiling/records/Deepfake/face2face/frames"
extract_frame_from_dir(source_root, des_root)
