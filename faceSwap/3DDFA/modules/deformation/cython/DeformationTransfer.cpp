#include <DeformationTransfer.h>
#include <algorithm>

#define EPSILON 0.0001

DeformationTransfer::DeformationTransfer()
{
    U = Uk = Ur = NULL;
    defVerts = NULL;
    bLog = false;
    regularization = 0;
    Reset();
}

DeformationTransfer::~DeformationTransfer()
{
    Reset();
}

void DeformationTransfer::SetRegularization(float regVal)
{
    regularization = regVal;
}

std::string DeformationTransfer::GetLastError()
{
    return strLastError;
}

void DeformationTransfer::Reset()
{
    nTris = nVerts = 0;
    if (U != NULL)    delete U;   U = NULL;
    if (Uk != NULL)   delete Uk;  Uk = NULL;
    if (Ur != NULL)   delete Ur;  Ur = NULL;
    if (defVerts != NULL)    delete defVerts;    defVerts = NULL;
}

void DeformationTransfer::SetLogging(bool bLogging)
{
    bLog = bLogging;
}

void DeformationTransfer::SetMatrixCol(MatrixXd &m, Vector3f v, int iCol)
{
    m(0, iCol) = v(0, 0);
    m(1, iCol) = v(1, 0);
    m(2, iCol) = v(2, 0);
}

void DeformationTransfer::SetTriEdgeMatrix(MatrixXd &Va, int nTris, const int *triVertIds, const float *pts, int iTri)
{
    int iV1 = triVertIds[iTri * 3 + 0];
    int iV2 = triVertIds[iTri * 3 + 1];
    int iV3 = triVertIds[iTri * 3 + 2];

    Vector3f v1(&pts[iV1 * 3]);
    Vector3f v2(&pts[iV2 * 3]);
    Vector3f v3(&pts[iV3 * 3]);
    
    Vector3f e1 = v2 - v1;
    Vector3f e2 = v3 - v1;
    Vector3f e3 = e1.cross(e2);
    float e3Len = e3.norm();
    if (e3Len > EPSILON)
    {
        e3 /= e3Len;
    }
 
    SetMatrixCol(Va, e1, 0);
    SetMatrixCol(Va, e2, 1);
    SetMatrixCol(Va, e3, 2);
}

void DeformationTransfer::SetMatrixBlock(MatrixXd &mBig, MatrixXd &mSmall, int iRow, int iCol)
{
    int r, c, rmax=mSmall.rows(), cmax=mSmall.cols();
    for (r=0; r<rmax; ++r)
    {
        for (c=0; c<cmax; ++c)
        {
            mBig(iRow + r, iCol + c) = mSmall(r, c);
        }
    }
}

void DeformationTransfer::SetSparseMatrixBlock(SparseMatrix<double> &mBig, MatrixXd &mSmall, int iRow, int iCol)
{
    int r, c, rmax=mSmall.rows(), cmax=mSmall.cols();
    for (r=0; r<rmax; ++r)
    {
        for (c=0; c<cmax; ++c)
        {
            mBig.coeffRef(iRow + r, iCol + c) = mSmall(r, c);
        }
    }
}

void DeformationTransfer::QRFactorize(const MatrixXd &a, MatrixXd &q, MatrixXd &r)
{
    int i, j, imax, jmax;
    imax = a.rows();
    jmax = a.cols();

    for (j=0; j<jmax; ++j)
    {
        Eigen::VectorXd v(a.col(j));
        for (i=0; i<j; ++i)
        {
            Eigen::VectorXd qi(q.col(i));
            r(i, j) = qi.dot(v);
            v = v - r(i, j) * qi;
        }
        float vv = (float)v.squaredNorm();
        float vLen = sqrtf(vv);
        if (vLen < EPSILON)
        {
            r(i, j) = 1;
            q.col(j).setZero();
        }
        else
        {
            r(i, j) = vLen;
            q.col(j) = v / vLen;
        }
    }
}

bool DeformationTransfer::SetSourceATargetA(int nTriangles, int nVertices, const int *triVertexIds, const float *vertexLocsSA, 
     const float *vertexLocsTA, ConstrainedVertIDs *vertConstrains)
{
    Reset();

    nTris = nTriangles;
    nVerts = nVertices;
    triVertIds = triVertexIds;
    vertLocsSA = vertexLocsSA;
    vertLocsTA = vertexLocsTA;

    MatrixXd Va(3, 2);

    U = new SparseMatrix<double>(3 * nTris, nVerts);
    defVerts = new float[3 * nVerts];

    int i, j;
    MatrixXd Uj(2, 3);
    for (j=0; j<nTris; ++j)
    {
        int iV1 = triVertIds[j * 3 + 0];  // point1 index, meshLab starts with 1
        int iV2 = triVertIds[j * 3 + 1];  // point2 index
        int iV3 = triVertIds[j * 3 + 2];  // point3 index

        Vector3f v1(&vertLocsTA[iV1 * 3]);  // point1, (x, y, z)
        Vector3f v2(&vertLocsTA[iV2 * 3]);  // point2, (x, y, z)
        Vector3f v3(&vertLocsTA[iV3 * 3]);  // point3, (x, y, z)

        SetMatrixCol(Va, v2 - v1, 0);
        SetMatrixCol(Va, v3 - v1, 1);
        MatrixXd Q(3, 2);
        MatrixXd R(2, 2);    R.setZero();
        QRFactorize(Va, Q, R);

        if (bLog)
        {
            std::cerr << "Matrix Va: \n" << Va << "\n\n";
            std::cerr << "Matrix Q:\n" << Q << "\n\n";
            std::cerr << "Matrix R:\n" << R << "\n\n";
        }

        R = R.inverse();  // (2, 2)
        Q.transposeInPlace();  // (2, 3)
        Uj = R * Q;  // (2, 3)

        if (bLog)
        {
            std::cerr << "Matrix Uj:\n" << Uj << "\n\n";
        }

        (*U).coeffRef(j * 3 + 0, iV1) = -Uj(0, 0) - Uj(1, 0);
        (*U).coeffRef(j * 3 + 0, iV2) = Uj(0, 0);
        (*U).coeffRef(j * 3 + 0, iV3) = Uj(1, 0);

        (*U).coeffRef(j * 3 + 1, iV1) = -Uj(0, 1) - Uj(1, 1);
        (*U).coeffRef(j * 3 + 1, iV2) = Uj(0, 1);
        (*U).coeffRef(j * 3 + 1, iV3) = Uj(1, 1);

        (*U).coeffRef(j * 3 + 2, iV1) = -Uj(0, 2) - Uj(1, 2);
        (*U).coeffRef(j * 3 + 2, iV2) = Uj(0, 2);
        (*U).coeffRef(j * 3 + 2, iV3) = Uj(1, 2);        
    }
    if (bLog)
    {
        std::cerr << "Matrix U:\n" << *U << "\n\n";
    }

    // constrain vertices
    Uk = new SparseMatrix<double>(3 * nTris, nVerts);
    int indxUk = 0;
    for (i=0; i<nVerts; ++i)
    {
        Uk->col(indxUk++) = U->col(i);
    }
    Ut = Uk->transpose();
    UtU = Ut * (*Uk);

    // Regularize matrix
    if (regularization != 0)
    {
        for (i=0; i<nVerts-1; ++i)
        {
            UtU.coeffRef(i, i) += regularization;
        }
    }

    // solve
    solver.compute(UtU);
    if (solver.info() != Eigen::Success)
    {
        strLastError = solver.lastErrorMessage();
        return false;
    }

    return true;
}

bool DeformationTransfer::Deformation(const float *vertLocsSB, const float *vertLocsTB, float **defVertLocs)
{
    MatrixXd *S = new MatrixXd(3 * nTris, 3);    S->setZero();

    MatrixXd Va(3, 3);
    MatrixXd Vb(3, 3);
    MatrixXd Sa(3, 3);

    int i;
    for (i=0; i<nTris; ++i)
    {
        SetTriEdgeMatrix(Va, nTris, triVertIds, vertLocsSA, i);
        SetTriEdgeMatrix(Vb, nTris, triVertIds, vertLocsSB, i);

        MatrixXd Q(3, 3);
        MatrixXd R(3, 3);    R.setZero();
        QRFactorize(Va, Q, R);

        if (bLog)
        {
            std::cerr << "Matrix Va:\n" << Va << "\n\n";
            std::cerr << "Matrix Q:\n" << Q << "\n\n";
            std::cerr << "Matrix R:\n" << R << "\n\n";
        }

        R = R.inverse();
        Q.transposeInPlace();
        Sa = Vb * R * Q;

        if (bLog)
        {
            std::cerr << "Matrix Vb:\n" << Vb << "\n\n";
            std::cerr << "Matrix Sa:\n" << Sa << "\n\n";
        }

        Sa.transposeInPlace();
        SetMatrixBlock(*S, Sa, i * 3, 0);
    }

    if (bLog)
    {
        std::cerr << "Matrix S:\n" << *S << "\n\n";
    }

    MatrixXd UtS = Ut * (*S);
 
    MatrixXd X = solver.solve(UtS);
    if (solver.info() != Eigen::Success)
    {
        strLastError = solver.lastErrorMessage();
        std::cout << strLastError << std::endl;
        return false;
    }

    delete S;

    float *currVert = defVerts;

    int indxX = 0;
    for (i=0; i<nVerts; ++i)
    {
        *(currVert++) = (float)X(indxX, 0);
        *(currVert++) = (float)X(indxX, 1);
        *(currVert++) = (float)X(indxX, 2);

        indxX++;
    }

    *defVertLocs = defVerts;
    return true;
}
