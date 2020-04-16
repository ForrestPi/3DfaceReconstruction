#ifndef MESH_CORE_HPP_
#define MESH_CORE_HPP_

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class point
{
    public:
        float x;
        float y;

        float dot(point p)
        {
            return this->x * p.x + this->y * p.y;
        }

        point operator-(const point& p)
        {
            point np;
            np.x = this->x - p.x;
            np.y = this->y - p.y;
            return np;
        }

        point operator+(const point& p)
        {
            point np;
            np.x = this->x + p.x;
            np.y = this->y + p.y;
            return np;
        }

        point operator*(const point& p)
        {
            point np;
            np.x = this->x * p.x;
            np.y = this->y * p.y;
            return np;
        }
};

bool isPointIntri(point p, point p0, point p1, point p2, int h, int w);

void get_point_weight(float* weight, point p, point p0, point p1, point p2);

void _render_colors_core(
    float* image, float* vertices, int* triangles, float* colors, float* depth_buffer, 
    int n_vertices, int n_triangles, int h, int w, int c);

#endif
