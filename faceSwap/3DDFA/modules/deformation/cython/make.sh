g++ -c deformation_core.cpp
g++ -c DeformationTransfer.cpp
g++ -o deformation deformation_core.o DeformationTransfer.o
./deformation
