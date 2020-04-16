#ifndef DEFORMATION_CORE_HPP_
#define DEFORMATION_CORE_HPP

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include "DeformationTransfer.h"

using namespace Eigen;

void _deformation_core(float* verticesSA, float* verticesTA, int* triVertexIds, \
     int nVertices, int nTriangles, float* verticesSBs, int nSBs, float* results);

#endif
