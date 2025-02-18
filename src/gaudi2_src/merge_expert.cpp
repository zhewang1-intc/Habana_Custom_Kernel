/**********************************************************************
Copyright (c) 2024 Habana Labs.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "merge_expert.hpp"

#include <cstring>
#include <iostream>
#include <vector>

extern unsigned char _binary___merge_expert_f32_o_start;
extern unsigned char _binary___merge_expert_f32_o_end;
extern unsigned char _binary___merge_expert_bf16_o_start;
extern unsigned char _binary___merge_expert_bf16_o_end;

tpc_lib_api::GlueCodeReturn MergeExpertGaudi2::GetKernelName(
    char kernelName[tpc_lib_api::MAX_NODE_NAME], merge_expert_mode mode) {
  if (mode == merge_expert_f32)
    strcpy(kernelName, "merge_expert_f32");
  else if (mode == merge_expert_bf16)
    strcpy(kernelName, "merge_expert_bf16");
  else
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn MergeExpertGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams *in_defs,
    tpc_lib_api::HabanaKernelInstantiation *out_defs) {
  tpc_lib_api::GlueCodeReturn retVal;

  unsigned token_num = in_defs->inputTensors[0].geometry.maxSizes[1];
  out_defs->indexSpaceRank = 1;
  out_defs->indexSpaceGeometry[0] = token_num;

  /*************************************************************************************
   *    Stage V -  Load ISA into the descriptor.
   **************************************************************************************/
  unsigned IsaSize =
      (&_binary___merge_expert_f32_o_end - &_binary___merge_expert_f32_o_start);
  unsigned char *binary_kernel;
  switch (mode) {
    case merge_expert_f32:
      IsaSize = (&_binary___merge_expert_f32_o_end -
                 &_binary___merge_expert_f32_o_start);
      binary_kernel = &_binary___merge_expert_f32_o_start;
      break;
    case merge_expert_bf16:
      IsaSize = (&_binary___merge_expert_bf16_o_end -
                 &_binary___merge_expert_bf16_o_start);
      binary_kernel = &_binary___merge_expert_bf16_o_start;
      break;
    default:
      break;
  }
  unsigned givenBinarySize = out_defs->kernel.elfSize;
  out_defs->kernel.elfSize = IsaSize;
  if (givenBinarySize >= IsaSize) {
    // copy binary out
    memcpy(out_defs->kernel.kernelElf, binary_kernel, IsaSize);
  } else {
    retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
    return retVal;
  }

  return tpc_lib_api::GLUE_SUCCESS;
}
