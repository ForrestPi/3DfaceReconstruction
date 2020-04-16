import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import dlib
import scipy.io as sio
import argparse
import math

from modules import mobilenet_v1
from utils import ToTensorGjz, NormalizeGjz, crop_img
from modules.morphable.morphable_model import MorphableModel
from modules import mesh

class Demo():
    def __init__(self, opt):
        self.opt = opt
        self.out_dir = opt.out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.load_model()

    def load_model(self):
        # 1. load trained model
        arch = 'mobilenet_1'
        state_dict = torch.load(self.opt.checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        new_state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict}
        self.model = getattr(mobilenet_v1, arch)(num_classes=self.opt.num_classes)
        model_dict = self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.eval()

        # 2. load dlib model for face detection and landmark used for face_cropping
        self.face_regressor = dlib.shape_predictor(self.opt.dlib_landmark_model)
        self.face_detector = dlib.get_frontal_face_detector()

        # 3. transform
        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

        # 4. load morphable_model
        self.morphable_model = MorphableModel(self.opt.morphable_model, self.opt.model_auxiliary)
        self.morphable_model.model_auxiliary['std_size'] = self.opt.std_size

        #for item in self.morphable_model.model['kpt_ind']:
        #    print(item)
        #for i in range(53215):
        #    if i not in self.morphable_model.model['kpt_ind']:
        #        print(i)
        #exit()

    def predict(self, img_path, dump_obj=None):
        dump_obj = dump_obj if dump_obj is not None else self.opt.dump_obj
        img_name = img_path.split('/')[-1].split('.')[0]
        img_origin = cv2.imread(img_path)
        rects = self.face_detector(img_origin, 1)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection
        vertices_list = []  # store multiple face vertices
        params_list = []
        roi_box_list = []
        colors_list = []

        for ind, rect in enumerate(rects):
            if self.opt.dlib_landmark:
                # - use landmark for roi box cropping
                pts = self.face_regressor(img_origin, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = self._parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = self._parse_roi_box_from_landmark(bbox)
            roi_box_list.append(roi_box)

            # step one
            img = crop_img(img_origin, roi_box)
            img = cv2.resize(img, dsize=(self.opt.std_size, self.opt.std_size), interpolation=cv2.INTER_LINEAR)
            img = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                img = img.cuda()
                params = self.model(img)
                params = params.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = self.morphable_model.predict_68pts(params, roi_box)

            # two-step for more acccurate bbox to crop face
            if self.opt.bbox_init == 'two':
                roi_box = self._parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_origin, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(self.opt.std_size, self.opt.std_size), interpolation=cv2.INTER_LINEAR)
                _img_step2 = img_step2.copy()
                img_step2 = self.transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    img_step2 = img_step2.cuda()
                    params = self.model(img_step2)
                    params = params.squeeze().cpu().numpy().flatten().astype(np.float32)
                    pts68 = self.morphable_model.predict_68pts(params, roi_box)

            params_list.append(params)

            vertices = self.morphable_model.predict_dense(params, roi_box)
            colors = mesh.transform.get_colors_from_image(img_origin, vertices) / 255.
            colors_list.append(colors)

            if dump_obj:
                path = os.path.join(self.out_dir, '{}_{}.obj'.format(img_name, ind))
                mesh.interact.write_obj_with_colors(path, vertices.T, self.morphable_model.model['tri'], colors)

                """
                h = img_origin.shape[0]
                w = img_origin.shape[1]
                image_vertices = vertices.copy().T
                #image_vertices[:, 1] = h - image_vertices[:, 1] - 1
                fitted_image = mesh.render.render_colors(image_vertices, self.morphable_model.triangles, colors, h, w) * 255.
                print(fitted_image.shape, image_vertices.shape, self.morphable_model.triangles.shape, colors.shape)
                cv2.imwrite(path.replace('obj', 'jpg'), fitted_image.astype('uint8'))
                """

        return params_list[0], colors_list[0], roi_box_list[0]

        #self.swap(*params_list, *colors_list, *roi_box_list, h, w)

    def swap1(self, params1, params2, colors1, colors2, roi_box1, roi_box2, h, w):
        params1_2 = params1.copy()
        params1_2[12: 52] = params2[12: 52]
        params2_1 = params2.copy()
        params2_1[12: 52] = params1[12: 52]

        vertices1_2 = self.morphable_model.predict_dense(params1_2, roi_box1).T
        vertices2_1 = self.morphable_model.predict_dense(params2_1, roi_box2).T
        image1_2 = mesh.render.render_colors(vertices1_2, self.morphable_model.triangles, colors2, h, w) * 255.
        image2_1 = mesh.render.render_colors(vertices2_1, self.morphable_model.triangles, colors1, h, w) * 255.

        cv2.imwrite('results/1_2.jpg', image1_2.astype('uint8'))
        cv2.imwrite('results/2_1.jpg', image2_1.astype('uint8'))

    def swap2(self, targetA, sourceBs, h=480, w=640):
        TAparams, TAcolors, TAroi_box = self.predict(targetA, dump_obj=False)
        BG = cv2.imread(targetA).astype(np.int32) / 255.
        for _root, _dirs, _files in os.walk(sourceBs):
            for _file in _files:
                sourceB = os.path.join(_root, _file)
                SBparams, SBcolors, SBroi_box = self.predict(sourceB, dump_obj=False)
                verticesSB = self.morphable_model.predict_dense(SBparams, TAroi_box).T
                Swapparams = SBparams.copy()
                Swapparams[12: 52] = TAparams[12: 52]
                Swapverices = self.morphable_model.predict_dense(Swapparams, TAroi_box).T
                #BG = cv2.imread(sourceB).astype(np.int32) / 255.
                Swapimage = mesh.render.render_colors(Swapverices, self.morphable_model.triangles, TAcolors, h, w, BG=BG) * 255.
                SBimage = mesh.render.render_colors(verticesSB, self.morphable_model.triangles, TAcolors, h, w) * 255.
                image = np.concatenate([SBimage, Swapimage], 1)
                print((SBimage != Swapimage).sum())
                img_name = sourceB.split('/')[-1]
                path = os.path.join(self.opt.out_dir, img_name.replace('obj', 'jpg'))
                cv2.imwrite(path, image.astype('uint8'))

    def swap(self, targets, sources, h=480, w=640):
        sources_list = []
        for _root, _dirs, _files in os.walk(sources):
            for _file in _files:
                source = os.path.join(_root, _file)
                sources_list.append(source)
        sources_list = sorted(sources_list)
        targets_list = []
        for _root, _dirs, _files in os.walk(targets):
            for _file in _files:
                target = os.path.join(_root, _file)
                targets_list.append(target)
        targets_list = sorted(targets_list)
        print(targets_list[0:10])
        print(sources_list[0:10])
        n = min(len(sources_list), len(targets_list))
        for i in range(n):
            source = sources_list[i]
            target = targets_list[i]
            source_name = source.split('/')[-2]
            target_name = target.split('/')[-2]
            ST_dir = os.path.join(self.opt.out_dir, '%s_%s' % (source_name, target_name))
            TS_dir = os.path.join(self.opt.out_dir, '%s_%s' % (target_name, source_name))
            if not os.path.exists(ST_dir):
                os.makedirs(ST_dir)
            if not os.path.exists(TS_dir):
                os.makedirs(TS_dir)
            BG_S = cv2.imread(source).astype(np.int32) / 255.
            BG_T = cv2.imread(target).astype(np.int32) / 255.
            Tparams, Tcolors, Troi_box = self.predict(target, dump_obj=False)
            Sparams, Scolors, Sroi_box = self.predict(source, dump_obj=False)
            STparams = Sparams.copy()
            STparams[52: ] = Tparams[52: ]
            TSparams = Tparams.copy()
            TSparams[52: ] = Sparams[52: ]
            STvertices = self.morphable_model.predict_dense(STparams, Sroi_box).T
            TSvertices = self.morphable_model.predict_dense(TSparams, Troi_box).T
            STimage = mesh.render.render_colors(STvertices, self.morphable_model.triangles, Scolors, h, w, BG=BG_S)
            TSimage = mesh.render.render_colors(TSvertices, self.morphable_model.triangles, Tcolors, h, w, BG=BG_T)
            STpath = os.path.join(ST_dir, "%04d.jpg" % i)
            TSpath = os.path.join(TS_dir, "%04d.jpg" % i)
            STimage = np.concatenate([BG_S, BG_T, STimage], 1) * 255.
            TSimage = np.concatenate([BG_T, BG_S, TSimage], 1) * 255.
            cv2.imwrite(STpath, STimage.astype('uint8'))
            cv2.imwrite(TSpath, TSimage.astype('uint8'))

               
    def _parse_roi_box_from_landmark(self, pts):
        """
        Args:
            pts: (2, n). n is the number of keypoints
        Returns:
            roi_box: list. (4, ). 4->(x1, y1, x2, y2)
        """
        bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]  # (x1, y1, x2, y2)

        llength = math.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
        center_x, center_y = center[0], center[1]

        roi_box = [0] * 4
        roi_box[0] = center_x - llength / 2
        roi_box[1] = center_y - llength / 2
        roi_box[2] = center_x + llength / 2
        roi_box[3] = center_y + llength / 2

        return roi_box


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('--file', default='/mnt/mfs/yiling/project/Deepfake/3DDFA/samples/1.jpg', type=str)
    #parser.add_argument('--file', default='results/test2.jpg', type=str)
    parser.add_argument('--std_size', default=120, type=int)
    parser.add_argument('--bbox_init', default='two', type=str)
    parser.add_argument('--num_classes', default=62, type=int)
    parser.add_argument('--checkpoint_fp', default='/mnt/mfs/yiling/project/Deepfake/3DDFA/models/phase1_wpdc_vdc.pth.tar', type=str)
    parser.add_argument('--dlib_landmark_model', default='/mnt/mfs/yiling/project/Deepfake/3DDFA/models/shape_predictor_68_face_landmarks.dat', type=str)
    parser.add_argument('--morphable_model', default='/mnt/mfs/yiling/GAN/Face3d/Data/BFM/Out/BFM.mat', type=str)
    parser.add_argument('--model_auxiliary', default='/mnt/mfs/yiling/project/Deepfake/3DDFA/utils/model_auxiliary.mat')
    #parser.add_argument('--out_dir', default='/mnt/mfs/yiling/records/Deepfake/face2face/fce_3dmm/003', type=str)
    parser.add_argument('--out_dir', default='output', type=str)
    parser.add_argument('--dump_obj', default=1, type=bool)
    parser.add_argument('--dlib_landmark', default=1, type=bool)

    opt = parser.parse_args()
    
    
    demo = Demo(opt)
    dir1 = "/mnt/mfs/yiling/records/Deepfake/face2face/face_frames/000"
    dir2 = "/mnt/mfs/yiling/records/Deepfake/face2face/face_frames/003"
    dir3 = "/mnt/mfs/yiling/project/Deepfake/Deep3DFaceReconstruction/input"
    for _root, _dirs, _files in os.walk(dir3):
        for _file in _files:
            path = os.path.join(_root, _file)
            if not path.endswith('png'):
                continue
            print(path)
            demo.predict(path)
    

    """
    opt.out_dir = "/mnt/mfs/yiling/records/Deepfake/face2face/swapdirect/000_003"
    print(opt)
    demo = Demo(opt)
    targetA = "/mnt/mfs/yiling/records/Deepfake/face2face/frames/003/003_0091.jpg"
    sourceBs = "/mnt/mfs/yiling/records/Deepfake/face2face/frames/000/"

    #targetA = "/mnt/mfs/yiling/records/Deepfake/face2face/frames/000/000_0000_0.jpg"
    #sourceBs = "/mnt/mfs/yiling/records/Deepfake/face2face/frames/003/"

    demo.swap(targetA, sourceBs)
    """

    """
    sources = "/mnt/mfs/yiling/records/Deepfake/face2face/frames/000/"
    targets = "/mnt/mfs/yiling/records/Deepfake/face2face/frames/003/"
    opt.out_dir = "/mnt/mfs/yiling/records/Deepfake/face2face/swapdirect"
    demo = Demo(opt)
    demo.swap(sources, targets)
    """
