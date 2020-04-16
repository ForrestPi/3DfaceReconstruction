import numpy as np

def load_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            files = pickle.load(f)
        return files
    else:
        print("{} not exist".format(filename))

class DeformationTransfer():
    def __init__(sourceAObj, targetAObj):
        self.SourceAVertices, self.SourceAColors, self.SourceATriangles = mesh.load_obj_with_colors(sourceAObj)
        self.TargetAVertices, self.TargetAColors, self.TargetATriangles = mesh.load_obj_with_colors(TargetAObj)

    def setSourceATargetA(self):
        tmean = 0
        tstd = np.sqrt(2)
        VS = self._normPts(self.SourceAVertices, tmean, tstd)
        VT = self._normPts(self.TargetAVertices, tmean, tstd)
        error, R, t, s = self._similarity_fitting(VT[self.marker[:, 1], :], VS[self.marker[:, 1])
        VT = VT.dot((s * R).T) + t
        if os.path.exists(file_name):
            VSP2 = load_pickle_file(file_name)
            return VSP2, VT
        else:
            TS, NS, VS4, FS4 = self._v4_normal(VS, FS)
            TT, NT, VT4, FT4 = self._v4_normal(VT, FT)
            Adj_idx = self._build_adjacency(FS)
            E = self._build_elementary_cell(TS, len(FT))
            M, C = self._build_phase1(Adj_idx, E, FS4, FT4, ws, wi, marker)

    def _build_adjacency(self, FS):
        """Build up the adjacency matrix
        Args:
            FS: (M, 3)
        Returns:
            Adj_idx: (M, 3), Adj_idx[i, j]
        """
        def func(Adj_idx, FS, i):
            for j in range(0, 3):
                idx = np.where(np.sum(FS == FS[i, j], axis=1) & \
                      np.sum(FS == FS[i, (j + 1) %3], axis=1))[0]
                if np.sum(idx != i):
                    Adj_idx[i, j] = int(idx[np.where(idx != i)])

        Adj_idx = np.zeros((FS.shape[0], 3), dtype=np.int32)
        Parallel(n_jobs=4, backend="threading")(delayed(func)(Adj_idx, FS, i) for i \
            in range(0, FS.shape[0]))
        return Adj_idx

    def _similarity_fitting(self, PointsA, PointsB):
        """calculate the R t s between PointsA and PointsB
        Args:
            PointsA: [n, 3] ndarray, n is the number of landmarks
            PointsB: [n, 3] ndarray
        Returns:
            res: Alignment error, (1, )
            R: (3, 3), rotation matrix
            t: (3, ),  translation vector
            s: (1, ), scale factor
        """
        row, col = PointsA.shape
        assert PointsA.shape == PointsB.shape
        PointsA = PointsA.T  # (3, n)
        pointsB = PointsB.T  # (3, n)
        cent = np.vstack((np.mean(PointsA, axis=1), np.mean(PointsB, axis=1))).T  #(3, 2)
        cent_0 = cent[:, 0][:, np.newaxis]  # (3, 1)
        cent_1 = cent[:, 1][:, np.newaxis]  # (3, 1)
        X = PointsA - cent_0  # (3, n)
        Y = PointsB - cent_1  # (3, n)
        #S = X.dot(np.eye(PointsA.shape[1], PointsA.shape[1])).dot(Y.T)  # (3, 3)
        S = X.dot(Y.T)  # (3, 3)
        U, D, V = np.linalg.svd(S) # (3, 3), (3, 3), (3, 3)
        V = V.T  # (3, 3)
        W = np.eye(V.shape[0], V.shape[0])  # (3, 3)
        W[-1, -1] = np.linalg.det(V.dot(U.T))  
        R = V.dot(W).dot(U.T)   # (3, 3)
        t = cent_1 - R.dot(cent_0)  # (3, 1)
        n = PointsA.shape[1]
        sigma2 = (1.0 / n) * np.multiply(cent_0, cent_0).sum()
        s = 1.0 / sigma2 * np.trace(np.dot(np.diag(D), W))
        b0 = np.zeros((8, ))
        if np.isreal(R).all():
            axis, theta = self._R_to_axis_angle(R)
            b0[0: 3] = axis
            b0[3] = theta
            if not np.isreal(b0).all():
                b0 = np.abs(b0)
        b0[4: 7] = t.T
        b0[7] = s
        b = least_squares(fun=self._resSimXform, x0=b0, jac='3-point', method='lm', args=(PointsA, PointsB), ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)
        r = b.x[0: 4]
        t = b.x[4: 7]
        s = b.x[7]
        R = self._axis_angle_to_R(r[0: 3], r[3])
        rotA = s * R.dot(PointsA) + t[:, np.newaxis]  # (3, n)
        res = np.sum(np.sqrt(np.sum((PointsB - rotA) ** 2, axis=1))) / PointsB.shape[1]
        print('Alignment error is {}'.format(res))
        return res, R, t, s

    def _R_to_axis_angle(self, R):
        """Convert the rotation matrix into the axis-angle notation.
           http://en.wikipedia.org/wiki/Rotation_matrix
               x = Qzy-Qyz
               y = Qxz-Qzx
               z = Qyx-Qxy
               r = hypot(x,hypot(y,z))
               t = Qxx+Qyy+Qzz
               theta = atan2(r,t-1)
        Args:
            R: 3x3 rotation matrix
        returns:
            axis: (3, )
            theta: (1, )
        """
        axis = np.zeros(3, np.float64)
        axis[0] = R[2, 1] - R[1, 2]
        axis[1] = R[0, 2] - R[2, 0]
        axis[2] = R[1, 0] - R[0, 1]
        r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
        t = R[0, 0] + R[1, 1] + R[2, 2]
        theta = math.atan2(r, t - 1)
        axis = axis / r
        return axis, theta

    def _axis_angle_to_R(self, axis, angle):
        """Generate the rotation matrix from the axis-angle notation.
           http://en.wikipedia.org/wiki/Rotation_matrix
               c = cos(angle); s = sin(angle); C = 1-c
               xs = x*s;   ys = y*s;   zs = z*s
               xC = x*C;   yC = y*C;   zC = z*C
               xyC = x*yC; yzC = y*zC; zxC = z*xC
               [ x*xC+c   xyC-zs   zxC+ys ]
               [ xyC+zs   y*yC+c   yzC-xs ]
               [ zxC-ys   yzC+xs   z*zC+c ]
        Args:
            axis: (3, )
            theta: (1, )
        Returns:
            R: (3, 3)
        """
        ca = np.cos(angle)
        sa = np.sin(angle)
        C = 1 - ca
        x, y, z = axis
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = y * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        R = np.zeros((3, 3), dtype=np.float64)
        R[0, 0] = x*xC + ca
        R[0, 1] = xyC - zs
        R[0, 2] = zxC + ys
        R[1, 0] = xyC + zs
        R[1, 1] = y*yC + ca
        R[1, 2] = yzC - xs
        R[2, 0] = zxC - ys
        R[2, 1] = yzC + xs
        R[2, 2] = z*zC + ca
        return R

    def _resSimXform(self, b, A, B):
        t = b[4: 7]
        R = self._axis_angle_to_R(b[0: 3], b[3])
        rotA = b[7] * R.dot(A) + t[:, np.newaxis]
        results = np.sqrt(np.sum((B - rotA) ** 2), axis=0)
        return results

    def _normPts(self, verts, mean, std):
        row, col = verts.shape
        T = np.eye(col + 1)  # (4, 4)
        mu = np.mean(verts, axis=0)  # (3, )
        T[0: col, col] = (mean - mu).T
        mean_distance = np.mean(np.sum(np.sqrt((verts - mu) ** 2), axis=1), axis=0)
        scale = std / mean_distance
        T = scale * T
        verts = np.concatenate((verts, np.ones((row, 1))), axis=1)  # (n, 4)
        verts = np.dot(T, verts.T).T  # (n, 4)
        verts = verts[:, 0: col]  # (n, 3)
        return verts

    def _v4_norm(self, Verts, Faces):
        """
        Args:
            Verts: (N, 3), ndarray
            Faces: (M, 3), ndarray
        Returns:
            T: (M)list, [(3, 3), ...]
            N: (M, 3), e4=v4-v1
            V: (N+M, 3), all v1, v2, v3 and v4(additional), new vertices
            F: (M, 4), triangle index(4 points), new faces
        """
        f1 = Faces[:, 0] - 1  #(M, ), Triangle index starts with 1
        f2 = Faces[:, 1] - 1  #(M, )
        f3 = Faces[:, 2] - 1  #(M, )
        e1 = Verts[f2, :] - Verts[f1, :]  # (M, 3)
        e2 = Verts[f3, :] - Verts[f1, :]  # (M, 3)
        c = np.cross(e1, e2)  # (M, 3)
        c_norm = np.sqrt(np.sum(c ** 2), axis=1) + 1e-16  # (M, )
        N = (c.T / c_norm).T  # (M, 3)
        v4 = Verts[f1, :] + N  # (M, 3)
        V = np.vstack((Verts, v4))  # (N+M, 3)
        F4 = Verts.shape[0] + np.where(Faces[:, 2])[0] + 1  # (M', ), M'=M, starts with N+1
        F = np.hstack((Faces, F4[:, np.newaxis]))  # (M, 4)
        T = []
        for i in range(0, F.shape[0]):
            Q = np.vstack((V[F[i, 1]-1, :] - V[F[i, 0]-1, :], \
                           V[F[i, 2]-1, :] - V[F[i, 0]-1, :], \
                           V[F[i, 3]-1, :] - V[F[i, 0]-1, :],))  # (3, 3)
            Q = np.transpose(Q) # (3, 3), (e1, e2, e3)
            T.append(Q)
        return T, N, V, F

    def _build_elementary_cell(self, T):
        def func(E, T, i):
            V = np.linalg.inv(T[i])
            E[i] = np.hstack((-np.sum(V, axis=0)[:, np.newaxis], V.T))  # (3, 4)

        m = len(T)
        E = [None for i in range(m)]
        Parallel(n_jobs=4, backend="threading")(delayed(func)(E, T, i) \
                for i range(m))
        return E
            
    def _build_phase1(Adj_idx, E, FS4, VT4, ws, wi, marker):
        """
        Args:
            Adj_idx: (M, 3)
            E: (M)list. [(3, 4), ...]
            FS4: (M, 4)
            VT4: (M, 4)
        """
        n_adj = Adj_idx.shape[0] * Adj_idx.shape[1]  # 3M
        len_col = np.max(FS4)
        I1 = np.zeros((9 * n_adj * 4, 3))
        I2 = np.zeros((9 * n_adj * 4, 3))
        I3 = np.zeros((9 * len(FS4) * 4, 3))  #(36M, 3)
        C1 = np.zeros((9 * n_adj, 1))
        C2 = wi * np.tile(np.reshape(np.eye(3), [9, 1]), (FS4.shape[0], 1))  #(9M, 1)
        for i in range(0, FS4.shape[0]):  # M
            for j in range(0, 3):
                if Adj_idx[i, j]:  # 3 lines of triangle
                    constid = np.zeros((2, 4))
                    for k in range(0, 3):
                        if np.sum(marker[:, 0] == FS4[i, k]):
                            constid[0, k] = (k + 1) * np.sum(marker[:, 0] == FS4[i, k])
                        if np.sum(marker[:, 0] == FS4[Adj_idx[i, j], k]):
                            constid[1, k] = (k + 1) * np.sum(marker[:, 0] == FS4[Adj_idx[i, j], k])
                    U1 = FS4[i, :]  # (4, )
                    U2 = FS4[Adj_idx[i, j], :]  # (4, )
                    for k in range(0, 3):
                        row = np.tile(np.linspace(0, 2, 3, dtype=np.int32) + i * 27 + \
                              j * 9 + k * 3, [4, 1])  # (4, 3)
                        col1 = np.tile((U1 - 1) * 3 + k, [3, 1]).T  # (4, 3)
