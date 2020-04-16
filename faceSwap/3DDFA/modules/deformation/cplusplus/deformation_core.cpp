/*
Create by: yiling
Create time: 2020,01,20 15:51
*/
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include "DeformationTransfer.h"

using namespace Eigen;

class TriangleMesh
{
public:
    std::vector<Vector3f> vertices;  // (n, 3)
    std::vector<Vector3i> triangles;  // (n, 3)
    std::vector<Vector3f> colors;   // (n, 3)

    int n_vertices;
    int n_triangles;

    template <typename T>
    T* flatten(std::vector< Matrix<T, 3, 1> > data, int n)
    {
        T* flatten_data = new T[3 * n]();
        int i = 0;
        for(typename std::vector< Matrix<T, 3, 1> >::iterator iter=data.begin(); iter!=data.end(); ++iter)
        {
            Matrix<T, 3, 1> tmp = (*iter);
            flatten_data[i++] = tmp[0];
            flatten_data[i++] = tmp[1];
            flatten_data[i] = tmp[2];
            if (i < 3 * n - 1)
                i++;
        }
        return flatten_data;
    }
};

bool load_obj(const std::string filename, TriangleMesh &mesh)
{
    std::ifstream in(filename.c_str());
    if (!in.good())
    {
        std::cerr << "ERROR: wrong obj file: " << filename << "\n";
        return false;
    }

    char buffer[256], str[256];
    int i_vertices = 0;
    int i_faces = 0;
    while (!in.getline(buffer, 255).eof())
    {
        buffer[255] = '\0';
        // reading a vertex
        if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32))
        {
            float f1 = 0, f2 = 0, f3 = 0;
            float c1 = 0, c2 = 0, c3 = 0;
            if (sscanf(buffer, "v %f %f %f %f %f %f", &f1, &f2, &f3, &c1, &c2, &c3) == 6)
            {
                Vector3f vertex(f1, f2, f3);
                mesh.vertices.push_back(vertex);
                Vector3f color(c1, c2, c3);
                mesh.colors.push_back(color);
                i_vertices++;
            }
            else
            {
                std::cerr << "ERROR: wrong vertex format in obj file" << '\n';
                return false;
            }
        }
        // reading a triangleface
        else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
        {
            int d1 = 0, d2 = 0, d3 = 0;
            if (sscanf(buffer, "f %d %d %d", &d1, &d2, &d3) == 3)
            {
                Vector3i face(d1 - 1, d2 - 1, d3 - 1);  // MeshLab starts with 1
                mesh.triangles.push_back(face);
                i_faces++;
            }
            else
            {
                std::cerr << "Error: wrong triangle face format in obj file" << '\n';
            }
        }
    }
    mesh.n_vertices = i_vertices;
    mesh.n_triangles = i_faces;
    return true;
}

void read_constrains_file(const std::string filePath, ConstrainedVertIDs &vertConstrains)
{
    std::ifstream in(filePath.c_str());
    char buffer[256];
    while (!in.getline(buffer, 255).eof())
    {
        buffer[255] = '\0';
        unsigned int ind = 0;
        if (sscanf(buffer, "%u", &ind) == 1)
        {
            vertConstrains.push_back(ind);
        }
    }
}

bool write_obj(const std::string filename, float *vertices, const int *triangles, const float *colors, const int nVertices, const int nTriangles)
{
    std::cout << "Dump file" << std::endl;
    std::ofstream obj_file(filename.c_str());
    for(int i=0; i<nVertices; ++i)
    {
        obj_file << "v " << *(vertices + 3*i) << " " << *(vertices + 3*i + 1) << " " << *(vertices + 3*i + 2) << " " << *(colors + 3*i) << " " << *(colors + 3*i + 1) << " " << *(colors + 3*i + 2) << std::endl;
    }

    for (int i=0; i<nTriangles; ++i)
    {
        obj_file << "f " << *(triangles + 3*i + 2) + 1 << " " << *(triangles + 3*i + 1) + 1 << " " << *(triangles + 3*i) + 1 << std::endl;
    }
    return true;
}

void deformation(TriangleMesh SA, TriangleMesh SB, TriangleMesh TA, TriangleMesh TB, const std::string constriansFile)
{
    DeformationTransfer DT;

    const int nTriangles = SA.n_triangles;
    const int nVertices = SA.n_vertices;
    const int *triVertexIds = SA.flatten(SA.triangles, nTriangles);
    const float *verticesSA = SA.flatten(SA.vertices, nVertices);
    const float *VerticesTA = TA.flatten(TA.vertices, nVertices);
    ConstrainedVertIDs vertConstrains;
    //read_constrains_file(constriansFile, vertConstrains);

    DT.SetSourceATargetA(nTriangles, nVertices, triVertexIds, verticesSA, VerticesTA, &vertConstrains);

    float *defVertLocs;
    const float *vertLocsSB = SB.flatten(SB.vertices, nVertices);
    const float *colorsTB = TB.flatten(TB.colors, nVertices);
    const float *vertLocsTB = TB.flatten(TB.vertices, nVertices);
    DT.Deformation(vertLocsSB, vertLocsTB, &defVertLocs);

    const std::string outFile = "out.obj";
    std::cout << "777777" << std::endl;
    write_obj(outFile, defVertLocs, triVertexIds, colorsTB, nVertices, nTriangles);
}

void read_filelist(const std::string filePath, std::vector<std::string> &pathlist)
{
    std::ifstream in(filePath.c_str());
    char buffer[256];
    while (!in.getline(buffer, 255).eof())
    {
        buffer[255] = '\0';
        std::string content = "";
        content = buffer;
        std::cout << content << std::endl;
        pathlist.push_back(content);
    }
}



void deformation_batch(TriangleMesh SA, const std::string SBfilelist, TriangleMesh TA, TriangleMesh TB, const std::string constriansFile)
{
    DeformationTransfer DT;

    const int nTriangles = SA.n_triangles;
    const int nVertices = SA.n_vertices;
    const int *triVertexIds = SA.flatten(SA.triangles, nTriangles);
    const float *verticesSA = SA.flatten(SA.vertices, nVertices);
    const float *VerticesTA = TA.flatten(TA.vertices, nVertices);
    ConstrainedVertIDs vertConstrains;
    //read_file(constriansFile, vertConstrains);
    std::vector<std::string> pathlist;
    read_filelist(SBfilelist, pathlist);

    DT.SetSourceATargetA(nTriangles, nVertices, triVertexIds, verticesSA, VerticesTA, &vertConstrains);

    for(std::vector<std::string>::iterator it = pathlist.begin(); it != pathlist.end(); ++it)
    {
        std::string SBpath = *it;
        TriangleMesh SB;
        load_obj(SBpath, SB);
        float *defVertLocs;
        const float *vertLocsSB = SB.flatten(SB.vertices, nVertices);
        const float *colorsTB = TB.flatten(TB.colors, nVertices);
        const float *vertLocsTB = TB.flatten(TB.vertices, nVertices);
        DT.Deformation(vertLocsSB, vertLocsTB, &defVertLocs);

        std::string outFile = SBpath;
        //outFile.replace(52, 3, "003_000");
        //outFile.replace(47, 4, "swap");
        outFile.replace(52+5, 3, "003_000");
        outFile.replace(47, 4+5, "face_swap");
        std::cout << "dump: " << outFile << std::endl;
        write_obj(outFile, defVertLocs, triVertexIds, colorsTB, nVertices, nTriangles);
    }
}


int main()
{   
    /*
    const std::string SAfilename = "/mnt/mfs/yiling/GAN/Face3d/face2face/3DDFA/results/001_f301_0.obj";
    const std::string SBfilename1 = "/mnt/mfs/yiling/GAN/Face3d/face2face/3DDFA/results/001_f269_0.obj";
    const std::string SBfilename2 = "/mnt/mfs/yiling/GAN/Face3d/face2face/3DDFA/results/001_f192_0.obj";
    const std::string SBfilename3 = "/mnt/mfs/yiling/GAN/Face3d/face2face/3DDFA/results/001_f14_0.obj";

    const std::string TAfilename = "/mnt/mfs/yiling/GAN/Face3d/face2face/3DDFA/results/000_f1_0.obj";
    TriangleMesh SA, SB1, SB2, SB3, TA;
    load_obj(SAfilename, SA);
    load_obj(SBfilename1, SB1);
    load_obj(SBfilename2, SB2);
    load_obj(SBfilename3, SB3);
    load_obj(TAfilename, TA);
    const std::string constrainsFile = "./constrains_ind.txt";
    deformation(SA, SB1, TA, TA, constrainsFile);
    */

    const std::string SAfilename = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/000/000_f0000_0.obj";
    const std::string TAfilename = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/003/003_f0091_0.obj";
    const std::string SBfilelist = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/000.txt";
    const std::string TBfilelist = "/mnt/mfs/yiling/records/Deepfake/face2face/face_3dmm/003.txt";
    TriangleMesh SA, TA;
    load_obj(SAfilename, SA);
    load_obj(TAfilename, TA);
    const std::string constrainsFile = "./constrains_ind.txt";
    
    //deformation_batch(SA, SBfilelist, TA, TA, constrainsFile);
    deformation_batch(TA, TBfilelist, SA, SA, constrainsFile);
}
