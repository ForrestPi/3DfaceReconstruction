import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

from scipy.io import loadmat
import numpy as np

import mesh

class Base3DMM(object):
    def __init__(self, model_path, n_shape_pca=None, n_exp_pca=None):
        self.model = self.load_BFM(model_path, n_shape_pca, n_exp_pca)

        self.n_vertices = int(self.model['shapePC'].shape[0] / 3)
        self.n_triangle = self.model['tri'].shape[0]
        self.n_shape_pca = self.model['shapePC'].shape[1]
        self.n_exp_pca = self.model['expPC'].shape[1]
        self.n_tex_pca = self.model['texPC'].shape[1]

        self.kpt_ind = self.model['kpt_ind']
        self.triangles = self.model['tri']
        self.full_triangles = np.vstack((self.model['tri'], self.model['tri_mouth']))

    def load_BFM(self, model_path, n_shape_pca=None, n_exp_pca=None):
        # suppport two kinds of format: pkl and mat
        # all the index start with 0
        if model_path.endswith('.pkl'):
            import pickle
            with open('model_path', 'rb') as f:
                model = pickle.load(f)
        elif model_path.endswith('.mat'):
            model = loadmat(model_path)

        model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
        model['shapePC'] = model['shapePC'].astype(np.float32) if n_shape_pca is None else model['shapePC'].astype(np.float32)[:, : n_shape_pca]
        model['shapeEV'] = model['shapeEV'].astype(np.float32) if n_shape_pca is None else model['shapeEV'].astype(np.float32)[:, : n_shape_pca]
        model['expPC'] = model['expPC'].astype(np.float32) if n_exp_pca is None else model['expPC'].astype(np.float32)[:, : n_exp_pca]
        model['expEV'] = model['expEV'].astype(np.float32) if n_exp_pca is None else model['expEV'].astype(np.float32)[:, : n_exp_pca]
        model['tri'] = model['tri'].astype(np.int32)
        model['tri_mouth'] = model['tri_mouth'].astype(np.int32)
        model['kpt_ind'] = (np.squeeze(model['kpt_ind'])).astype(np.int32)

        print('====> Load model from %s successfully' % model_path)
        print('###### BFM infomation')
        for key in model:
            try:
                print('BFM[%s]' % key, model[key].shape)
            except:
                continue
        return model

    def get_shape_params(self, _type='random'):
        if _type == 'zero':
            sp = np.random.zeros((self.n_shape_pca, 1))
        elif _type == 'random':
            sp = np.random.random([self.n_shape_pca, 1]) * 1e04   # why 1e04 ???
        return sp

    def get_exp_params(self, _type='random'):
        if _type == 'zero':
            ep = np.zeros((self.n_exp_pca, 1))
        elif _type == 'random':
            ep = -1.5 + 3 * np.random.random([self.n_exp_pca, 1])
            ep[6:, 0] = 0
        return ep

    def generate_vertices(self, shape_params, exp_params):
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(shape_params) + self.model['expPC'].dot(exp_params)
        vertices = np.reshape(vertices, [3, self.n_vertices], 'F').T  # (n_vertices, 3)
        return vertices

    def get_tex_params(self, _type='random'):
        if _type == 'zero':
            tp = np.zeros((self.n_tex_pca, 1))
        elif _type == 'random':
            tp = np.random.rand(self.n_tex_pca, 1)
        return tp

    def generate_colors(self, tex_params):
        colors = self.model['texMU'] + self.model['texPC'].dot(tex_params * self.model['texEV'])
        colors = np.reshape(colors, [3, self.n_vertices], 'F').T / 255.  # (n_vertices, 3)
        return colors

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)
