#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/SparseQR>

using Eigen::DynamicSparseMatrix;
using Eigen::SparseMatrix;
using Eigen::MatrixXd;
using Eigen::HouseholderQR;
using Eigen::Vector3d;
using Eigen::Vector3f;

typedef std::vector<unsigned int> ConstrainedVertIDs;

class DeformationTransfer
{
public:
    DeformationTransfer();
    ~DeformationTransfer();

    bool SetSourceATargetA(int nTriangles, int nVertices, const int* triVertexIds, const float* vertexLocsSA, 
         const float* vertexLocsTA, ConstrainedVertIDs* vertConstrains=NULL);

    bool Deformation(const float* verLocsSB, const float* vertLocsTB, float** defVertLocs);

    std::string GetLastError();

    void SetLogging(bool bLogging);

    void SetRegularization(float regVal);

protected:

private:
    int nTris, nVerts;
    const int* triVertIds;
    const float *vertLocsSA, *vertLocsTA;
    float *defVerts;

    SparseMatrix<double> *U, *Uk, *Ur, Ut, UtU;

    std::string strLastError;

    bool bLog;
    float regularization;

    ConstrainedVertIDs constrainedVertIDs;
    Eigen::SparseLU< SparseMatrix<double> > solver;

    void SetMatrixBlock(MatrixXd &mBig, MatrixXd &mSmall, int iRow, int iCol);
    void SetSparseMatrixBlock(SparseMatrix<double> &mBig, MatrixXd &mSmall, int iRow, int iCol);
    void SetMatrixCol(MatrixXd &m, Vector3f v, int iCol);
    void SetTriEdgeMatrix(MatrixXd &Va, int nTriangles, const int *triVertIds, const float *pts, int iTriangle);
    void Reset();

    void QRFactorize(const MatrixXd &a, MatrixXd &q, MatrixXd &r);
};
