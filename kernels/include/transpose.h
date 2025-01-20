/*****************************************************************************
 * Copyright (C) 2019 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Dmytro Salnikov <dmytro.salnikov@p-product.com>
 ******************************************************************************
 */
#include "kernel_config.h"
#pragma tpc_printf(enable)

enum
{
    depth = 0,
    width = 1,
    height = 2,
    batch = 3,
    fifthDim = 4
};

#define INDEX_SPACE_HEAD(d, w, h, b)                                         \
    const int5 index_space_start = get_index_space_offset();                 \
    const int5 index_space_end = get_index_space_size() + index_space_start; \
    const int depthStep = d;                                                 \
    const int depthStart = index_space_start[depth] * depthStep;             \
    const int depthEnd = index_space_end[depth] * depthStep;                 \
    const int widthStep = w;                                                 \
    const int widthStart = index_space_start[width] * widthStep;             \
    const int widthEnd = index_space_end[width] * widthStep;                 \
    const int heightStep = h;                                                \
    const int heightStart = index_space_start[height] * heightStep;          \
    const int heightEnd = index_space_end[height] * heightStep;              \
    const int batchStep = b;                                                 \
    const int batchStart = index_space_start[batch] * batchStep;             \
    const int batchEnd = index_space_end[batch] * batchStep;

void main(
    tensor ifm,
    tensor ofm,

    int axes0,
    int axes1,
    int axes2,
    int axes3)
{
    INDEX_SPACE_HEAD(1, 1, 1, 1);

    int5 coordsIn = {0};
    int5 coordsOut = {0, 0, 0, 0, 0};

    int ofmDepth = get_dim_size(ofm, depth);
    int ofmWidth = get_dim_size(ofm, width);
    int ofmHeight = get_dim_size(ofm, height);
    int ofmBatch = get_dim_size(ofm, batch);

    int ofmDepthStride = get_dim_stride(ofm, depth);
    int ofmWidthStride = get_dim_stride(ofm, width);
    int ofmHeightStride = get_dim_stride(ofm, height);
    int ofmBatchStride = get_dim_stride(ofm, batch);

    // make tensor 1 5d
    int ofmConfig = get_tensor_config(ofm);
    int ofmNewConfig = config_add_one_dim(ofmConfig);
    set_tensor_config(ofm, ofmNewConfig);

    // set fcd = 1 and move other dimansions to use vector load/stores
    set_dim_size(ofm, 0, 1);
    set_dim_stride(ofm, 0, 1);

    set_dim_size(ofm, 1, ofmDepth); // 4
    set_dim_stride(ofm, 1, ofmDepthStride);

    set_dim_size(ofm, 2, ofmWidth); // 1
    set_dim_stride(ofm, 2, ofmWidthStride);

    set_dim_size(ofm, 3, ofmHeight); // 3
    set_dim_stride(ofm, 3, ofmHeightStride);

    set_dim_size(ofm, 4, ofmBatch); // 2
    set_dim_stride(ofm, 4, ofmBatchStride);
    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        coordsIn[0] = d;

        // coordsOut[axes[0]] = d;
        coordsOut = set_indx(d, coordsOut, 2, 0, axes0 == 0);
        coordsOut = set_indx(d, coordsOut, 4, 0, axes0 == 1);
        coordsOut = set_indx(d, coordsOut, 8, 0, axes0 == 2);
        coordsOut = set_indx(d, coordsOut, 16, 0, axes0 == 3);

        for (int b = batchStart; b < batchEnd; b += batchStep)
        {
            coordsIn[3] = b;

            // coordsOut[axes[3]] = b;
            coordsOut = set_indx(b, coordsOut, 2, 0, axes3 == 0);
            coordsOut = set_indx(b, coordsOut, 4, 0, axes3 == 1);
            coordsOut = set_indx(b, coordsOut, 8, 0, axes3 == 2);
            coordsOut = set_indx(b, coordsOut, 16, 0, axes3 == 3);

            for (int h = heightStart; h < heightEnd; h += heightStep)
            {
                coordsIn[2] = h;

                // coordsOut[axes[2]] = h;
                coordsOut = set_indx(h, coordsOut, 2, 0, axes2 == 0);
                coordsOut = set_indx(h, coordsOut, 4, 0, axes2 == 1);
                coordsOut = set_indx(h, coordsOut, 8, 0, axes2 == 2);
                coordsOut = set_indx(h, coordsOut, 16, 0, axes2 == 3);

                for (int w = widthStart; w < widthEnd; w += widthStep)
                {
                    coordsIn[1] = w;

                    // coordsOut[axes[1]] = w;
                    coordsOut = set_indx(w, coordsOut, 2, 0, axes1 == 0);
                    coordsOut = set_indx(w, coordsOut, 4, 0, axes1 == 1);
                    coordsOut = set_indx(w, coordsOut, 8, 0, axes1 == 2);
                    coordsOut = set_indx(w, coordsOut, 16, 0, axes1 == 3);

                    __global__ void *addrIn = gen_addr(coordsIn, ifm);

                    VECTOR x = v_ld_g_a(addrIn);
                    printf("%f\n", x[0]);
                    st_tnsr_i_v(coordsOut, ofm, x);
                }
            }
        }
    }
}
