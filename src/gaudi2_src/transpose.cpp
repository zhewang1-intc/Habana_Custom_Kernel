/**********************************************************************
Copyright (c) 2024 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <vector>
#include <cstring>
#include <iostream>
#include "transpose.hpp"

extern unsigned char _binary___transpose_f32_o_start;
extern unsigned char _binary___transpose_f32_o_end;
extern unsigned char _binary___transpose_bf16_o_start;
extern unsigned char _binary___transpose_bf16_o_end;

tpc_lib_api::GlueCodeReturn TransposeGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME], Transpose_mode mode)
{
    if (mode == transpose_f32)
        strcpy(kernelName, "transpose_f32");
    else if (mode == transpose_bf16)
        strcpy(kernelName, "transpose_bf16");
    else
        return tpc_lib_api::GLUE_NODE_NOT_FOUND;
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn TransposeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams *in_defs,
    tpc_lib_api::HabanaKernelInstantiation *out_defs)
{
    // TODO: add tensor validate.
    tpc_lib_api::GlueCodeReturn retVal;

    uint64_t *inputSizes = in_defs->inputTensors[0].geometry.maxSizes;

    for (int i = 0; i < 4; i++)
    {
        out_defs->indexSpaceGeometry[i] = std::max(inputSizes[i], 1lu);
    }

    out_defs->inputTensorAccessPattern[0].mapping[0].a = 1;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b = 0;

    out_defs->outputTensorAccessPattern[0].mapping[0].a = 1;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b = 0;

    out_defs->outputTensorAccessPattern[0].allRequired = 1;

    /*************************************************************************************
     *    Stage IV -  define scalar parameters
     **************************************************************************************/
    TpcTransParams *p = static_cast<TpcTransParams *>(in_defs->nodeParams.nodeParams);
    out_defs->kernel.paramsNr = sizeof(*p) / sizeof(float);
    memcpy(&(out_defs->kernel.scalarParams[0]), p, sizeof(*p));

    /*************************************************************************************
     *    Stage V -  Load ISA into the descriptor.
     **************************************************************************************/
    unsigned IsaSize = (&_binary___transpose_f32_o_end - &_binary___transpose_f32_o_start);
    unsigned char *binary_kernel;
    switch (sdpa_mode)
    {
    case transpose_f32:
        IsaSize = (&_binary___transpose_f32_o_end - &_binary___transpose_f32_o_start);
        binary_kernel = &_binary___transpose_f32_o_start;
        break;
    case transpose_bf16:
        IsaSize = (&_binary___transpose_bf16_o_end - &_binary___transpose_bf16_o_start);
        binary_kernel = &_binary___transpose_bf16_o_start;
        break;
    default:
        break;
    }
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;
    if (givenBinarySize >= IsaSize)
    {
        // copy binary out
        memcpy(out_defs->kernel.kernelElf,
               binary_kernel,
               IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
