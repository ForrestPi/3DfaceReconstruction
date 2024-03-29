# BFM模型基本介绍
Basel Face Model是一个开源的人脸 数据库，其基本原理是3DMM，因此其便是在PCA的基础上进行存储的。 目前有两个版本的数据库（2009和2017）。 官方网站：2009，2017
##  数据内容（以2009版本为例）
## 文件内容
01_MorphableModel.mat（数据主体）

BFM模型由53490个顶点构成，其shape/texture的数据长度为160470（53490*3），因为其排列方式如下：

    shape: x_1, y_1, z_1, x_2, y_2, z_2, ..., x_{53490}, y_{53490}, z_{53490}
    texture: r_1, g_1, b_1, r_2, g_2, b_2, ..., r_{53490}, g_{53490}, b_{53490}

.h5文件与.mat文件对应关系

[注] .h5文件中的tl数量与.mat数量不同，主成分方差的值也不同，且shape的值是.mat中shape值的0.001倍（见/shape/representer/length-unit）。

* shapeEV：形状方差；
* shapeMU（160470*1）：平均形状；
* shapePC：形状的主成分；
* texEV：纹理方差；
* texMU：平均纹理
* texPC：纹理的主成分；
