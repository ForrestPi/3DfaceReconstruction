/*
Create by: yiling
Create time: 2020,01,20 15:51
*/
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <ctime>
#include "DeformationTransfer.h"

using namespace Eigen;

void _deformation_core(float* verticesSA, float* verticesTA, int* triVertexIds, int nVertices, int nTriangles, \
     float* verticesSBs, int nSBs, float* results)
{
    DeformationTransfer DT;
    ConstrainedVertIDs vertConstrains;
    time_t t1 = time(0);
    std::cout << "DeformatinTransfer initialize start: " << t1 << std::endl;
    DT.SetSourceATargetA(nTriangles, nVertices, triVertexIds, verticesSA, verticesTA, &vertConstrains);
    time_t t2 = time(0);
    std::cout << "DeformatinTransfer initialization consume: " << t2 - t1 << " seconds" << std::endl;
    for(int i=0; i<nSBs; ++i)
    {
        float* verticesSB = new float[3 * nVertices]();
        for(int j=0; j<3 * nVertices; ++j)
        {
            *(verticesSB + j) = *(verticesSBs + i * 3 * nVertices + j);
        }
        float *defVertLocs;
        time_t t3 = time(0);
        DT.Deformation(verticesSB, verticesTA, &defVertLocs);
        time_t t4 = time(0);
        std::cout << i << " deformatin consume: " << t4 - t3 << " seconds" << std::endl;
        for(int j=0; j<3 * nVertices; ++j)
        {
            *(results + i * 3 * nVertices + j) = *(defVertLocs + j);
        }
    }
}





