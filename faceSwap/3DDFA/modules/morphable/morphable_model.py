import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
from scipy.io import loadmat

from base_3dmm import Base3DMM

class MorphableModel(Base3DMM):
    def __init__(self, model_path, model_auxiliary_path, n_shape_pca=40, n_exp_pca=10):
        super(MorphableModel, self).__init__(model_path, n_shape_pca, n_exp_pca)
        self.model_auxiliary = loadmat(model_auxiliary_path)

    def _parse_params(self, params, whitening=True, affine=True):
        if len(params) == 12:
            params = np.concatenate((params, [0] * 50))
        if whitening:
            if len(params) == 62:
                params = params * self.model_auxiliary['param_std'][0] + self.model_auxiliary['param_mean'][0]
            else:
                params = np.concatenate((params[: 11], [0], params[11: ]))
                params = params * self.model_auxiliary['param_std'] + self.model_auxiliary['param_mean']

        p_ = params[: 12].reshape(3, -1)
        p = p_[:, :3]
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = params[12: 52].reshape(-1, 1)
        alpha_exp = params[52: ].reshape(-1, 1)

        if not affine:
            p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            offset = np.array([[0], [0], [0]], dtype=np.float32)

        return p, offset, alpha_shp, alpha_exp

    def reconstruct_vertices(self, params, whitening=True, dense=False, transform=True, affine=True):
        p, offset, alpha_shp, alpha_exp = self._parse_params(params, whitening=whitening, affine=affine)

        if dense:
            vertices = p @ (self.model['shapeMU'] + self.model['shapePC'] @ alpha_shp + self.model['expPC'] @ alpha_exp).\
                       reshape(3, -1, order='F') + offset
        else:
            ind = self.model_auxiliary['kpt68_ind_x3'].squeeze()
            vertices = p @ (self.model['shapeMU'][ind] + self.model['shapePC'][ind] @ alpha_shp + self.model['expPC'][ind] @ alpha_exp).\
                       reshape(3, -1, order='F') + offset

        if transform:
            vertices[1, :] = self.model_auxiliary['std_size'] + 1 - vertices[1, :]

        return vertices

    def predict_vertices(self, params, roi_box, dense, transform=True, affine=True):
        vertices = self.reconstruct_vertices(params, dense=dense, affine=affine)
        return self._shift_vertices(vertices, roi_box)

    def _shift_vertices(self, vertices, roi_box):
        sx, sy, ex, ey = roi_box
        scale_x = (ex - sx) / self.model_auxiliary['std_size']
        scale_y = (ey - sy) / self.model_auxiliary['std_size']
        vertices[0, :] = vertices[0, :] * scale_x + sx
        vertices[1, :] = vertices[1, :] * scale_y + sy

        scale_z = (scale_x + scale_y) / 2
        vertices[2, :] *= scale_x
        return vertices

    def predict_68pts(self, params, roi_box, affine=True):
        return self.predict_vertices(params, roi_box, dense=False, affine=affine)

    def predict_dense(self, params, roi_box, affine=True):
        return self.predict_vertices(params, roi_box, dense=True, affine=affine)
