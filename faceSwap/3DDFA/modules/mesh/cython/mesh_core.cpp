
#include "mesh_core.h"

void get_point_weight(float* weight, point p, point p0, point p1, point p2)
{
    point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    float dot00 = v0.dot(v0); // v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); // v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); // v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); // v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); // v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if(dot00 * dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}

bool isPointInTri(point p, point p0, point p1, point p2)
{
    point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    float dot00 = v0.dot(v0); // v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); // v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); // v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); // v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); // v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if(dot00 * dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // check if point p in triangle
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

void _render_colors_core(float* image, float* vertices, int* triangles, float* colors, float* depth_buffer,
                         int n_vertices, int n_triangles, int h, int w, int c)
{
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float p_color, p0_color, p1_color, p2_color;
    float weight[3];

    for(i = 0; i < n_triangles; ++i)
    {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        p0.x = vertices[3 * tri_p0_ind]; p0.y = vertices[3 * tri_p0_ind + 1]; p0_depth = vertices[3 * tri_p0_ind + 2];
        p1.x = vertices[3 * tri_p1_ind]; p1.y = vertices[3 * tri_p1_ind + 1]; p1_depth = vertices[3 * tri_p1_ind + 2];
        p2.x = vertices[3 * tri_p2_ind]; p2.y = vertices[3 * tri_p2_ind + 1]; p2_depth = vertices[3 * tri_p2_ind + 2];

        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if((x_max < x_min) || (y_max < y_min))
        {
            continue;
        }

        for(y = y_min; y <= y_max; ++y)
        {
            for(x = x_min; x <= x_max; ++x)
            {
                p.x = x;
                p.y = y;
                if(p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3 || isPointInTri(p, p0, p1, p2))
                {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if(p_depth > depth_buffer[y * w + x])
                    {
                        for(k = 0; k < c; ++k) // c
                        {
                            p0_color = colors[c * tri_p0_ind + k];
                            p1_color = colors[c * tri_p1_ind + k];
                            p2_color = colors[c * tri_p2_ind + k];

                            p_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color;
                            image[y * w * c + x * c + k] = p_color;
                        }
                        depth_buffer[y * w + x] = p_depth;
                    }
                }
            } // x, w
        } // y, h
    } // i
}
