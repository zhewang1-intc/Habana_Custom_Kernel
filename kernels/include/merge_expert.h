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

#define UNROLL_LOAD 4

void main(tensor input, tensor indices, tensor weight, tensor output) {
  const int5 token_start = get_index_space_offset();
  const int5 token_end = get_index_space_size() + token_start;
  const int dim = get_dim_size(input, 0);
  const int select_expert_num = get_dim_size(indices, 0);
  int5 input_coords[UNROLL_LOAD] = {0};
  int5 output_coords[UNROLL_LOAD] = {0};
  int5 info_coords = {0};
  VECTOR weight_reg;
  VECTOR in_reg[UNROLL_LOAD];
  VECTOR out_reg[UNROLL_LOAD];
  for (int cur_token = token_start[0]; cur_token < token_end[0]; cur_token++) {
#pragma unroll(UNROLL_LOAD)
    for (int i = 0; i < UNROLL_LOAD; i++) {
      input_coords[i][1] = cur_token;
      output_coords[i][1] = cur_token;
    }
    info_coords[1] = cur_token;
    for (int expert_id = 0; expert_id < select_expert_num; expert_id++) {
      info_coords[0] = expert_id;
      weight_reg = v_ld_g_a(gen_addr(info_coords, weight));
      __global__ int* select_expert_id_ptr = gen_addr(info_coords, indices);
      int select_expert_id = s_i32_ld_g(select_expert_id_ptr);
#pragma unroll(UNROLL_LOAD)
      for (int i = 0; i < UNROLL_LOAD; i++)
        input_coords[i][2] = select_expert_id;
      for (int cur_dim = 0; cur_dim < dim;
           cur_dim += VECTOR_SIZE * UNROLL_LOAD) {
#pragma unroll(UNROLL_LOAD)
        for (int i = 0; i < UNROLL_LOAD; i++) {
          input_coords[i] = cur_dim + i * VECTOR_SIZE;
          output_coords[i] = cur_dim + i * VECTOR_SIZE;
          in_reg[i] = v_ld_tnsr_i(input_coords, input);
          out_reg[i] = v_ld_tnsr_i(output_coords, output);
          out_reg[i] = v_mac_v_v(in_reg[i], weight_reg, out_reg[i]);
          st_tnsr_i_v(output_coords, output, out_reg[i]);
        }
      }
    }
  }
}
