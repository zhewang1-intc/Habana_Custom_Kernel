/**********************************************************************
Copyright (c) 2024 Habana Labs. All rights reserved.

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
#include "kernel_config.h"
#pragma tpc_printf(enable)

#define broadcast_unroll 4  // should same as repeat_kv_num.
#define NO_SLM_SOFTMAX

#define PRINT_REG_VALUE(NAME, REG)                    \
  printf(NAME);                                       \
  for (int i = 0; i < 64; i++) printf("%f ", REG[i]); \
  printf("\n");

void dump_max(bfloat128 reg) {
  float128 t = convert_bfloat128_to_float128(reg, SW_RHNE);
  printf("QK_max %f\n", t.v1[0]);
}

bfloat128 bf16_exp(bfloat128 reg) {
  float128 tmp_128 = convert_bfloat128_to_float128(reg, SW_RHNE);
  tmp_128.v1 = v_exp_f32(tmp_128.v1);
  tmp_128.v2 = v_exp_f32(tmp_128.v2);
  reg = convert_float128_to_bfloat128(tmp_128, SW_RHNE);
  return reg;
}

// #define bf16_div(reg)                                        \
//       tmp_128 = convert_bfloat128_to_float128(reg, SW_RHNE); \
//       tmp_128.v1 = v_div_f32(tmp, tmp_128.v1);               \
//       tmp_128.v2 = v_div_f32(tmp, tmp_128.v2);               \
//       reg = convert_float128_to_bfloat128(tmp_128, SW_RHNE);

bfloat128 bf16_div(bfloat128 reg, float64 tmp) {
  float128 tmp_128 = convert_bfloat128_to_float128(reg, SW_RHNE);
  tmp_128.v1 = v_div_f32(tmp, tmp_128.v1);
  tmp_128.v2 = v_div_f32(tmp, tmp_128.v2);
  reg = convert_float128_to_bfloat128(tmp_128, SW_RHNE);
  return reg;
}

#define set_coords(coords, cur_head)              \
  coords[0][2] = cur_head * broadcast_unroll;     \
  coords[1][2] = cur_head * broadcast_unroll + 1; \
  coords[2][2] = cur_head * broadcast_unroll + 2; \
  coords[3][2] = cur_head * broadcast_unroll + 3;

__local__ VECTOR slm_test[1024 * 16 / 4 / 64];
void main(tensor Q, tensor K, tensor QK, tensor V, tensor Out) {
  const int5 K_head_start = get_index_space_offset();
  const int5 K_head_end = get_index_space_size() + K_head_start;

  const int head_dim = get_dim_size(Q, 0);
  const int q_seq_len = get_dim_size(Q, 1);
  const int kv_seq_len = get_dim_size(K, 0);
  const int q_head_num = get_dim_size(Q, 2);
  const int kv_head_num = get_dim_size(K, 2);
  const int repeat_kv_num = q_head_num / kv_head_num;

  printf("k_start %d\n", K_head_start[0]);
  printf("k_end %d\n", K_head_end[0]);
  printf("kv seq len %d\n", kv_seq_len);

  int5 Q_coords = {0};
  int5 K_coords = {0};
  int5 QK_coords[4];
  int5 QK_coords2[4];
  int5 V_coords = {0};
  //   int5 Out_coords = {0};
  int5 Out_coords[4];

  QK_coords[0][1] = 0;
  QK_coords[1][1] = 0;
  QK_coords[2][1] = 0;
  QK_coords[3][1] = 0;

  VECTOR sqrt_dk = 2.f;  // todo: make it as param.

  VECTOR broadcast_reg[broadcast_unroll];
  VECTOR KV_reg;
  VECTOR acc_reg[broadcast_unroll];
  VECTOR QK_max[broadcast_unroll];
  VECTOR QK_exp_sum[broadcast_unroll];
  float64 tmp = 1.f;
  float128 tmp_128;

  for (int cur_k_head = K_head_start[0]; cur_k_head < K_head_end[0]; cur_k_head++) {
    K_coords[2] = cur_k_head;
    V_coords[2] = cur_k_head;
    // gemv
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq += VECTOR_SIZE) {
      K_coords[0] = cur_kv_seq;
#pragma loop_taken
      for (int i = 0; i < broadcast_unroll; i++) {
        acc_reg[i] = 0.f;
        QK_max[i] = 0.f;
      }
      for (int broadcast_Q_dim = 0; broadcast_Q_dim < head_dim; broadcast_Q_dim++) {
        Q_coords[2] = cur_k_head * broadcast_unroll;
        Q_coords[0] = broadcast_Q_dim;
        __global__ float* p_Q0 = gen_addr(Q_coords, Q);
        Q_coords[2] += 1;
        __global__ float* p_Q1 = gen_addr(Q_coords, Q);
        Q_coords[2] += 1;
        __global__ float* p_Q2 = gen_addr(Q_coords, Q);
        Q_coords[2] += 1;
        __global__ float* p_Q3 = gen_addr(Q_coords, Q);
        broadcast_reg[0] = v_ld_g_a(p_Q0);
        broadcast_reg[1] = v_ld_g_a(p_Q1);
        broadcast_reg[2] = v_ld_g_a(p_Q2);
        broadcast_reg[3] = v_ld_g_a(p_Q3);
        K_coords[1] = broadcast_Q_dim;
        KV_reg = v_ld_tnsr_i(K_coords, K);
        acc_reg[0] = v_mac_v_v(KV_reg, broadcast_reg[0], acc_reg[0]);
        acc_reg[1] = v_mac_v_v(KV_reg, broadcast_reg[1], acc_reg[1]);
        acc_reg[2] = v_mac_v_v(KV_reg, broadcast_reg[2], acc_reg[2]);
        acc_reg[3] = v_mac_v_v(KV_reg, broadcast_reg[3], acc_reg[3]);
      }

      QK_max[0] = v_max_v_v(acc_reg[0], QK_max[0]);
      QK_max[1] = v_max_v_v(acc_reg[1], QK_max[1]);
      QK_max[2] = v_max_v_v(acc_reg[2], QK_max[2]);
      QK_max[3] = v_max_v_v(acc_reg[3], QK_max[3]);
      QK_coords[0][0] = cur_kv_seq;
      QK_coords[1][0] = cur_kv_seq;
      QK_coords[2][0] = cur_kv_seq;
      QK_coords[3][0] = cur_kv_seq;

      // set_coords(QK_coords, cur_k_head);
      // st_tnsr_i_v(QK_coords[0], QK, acc_reg[0]);
      // st_tnsr_i_v(QK_coords[1], QK, acc_reg[1]);
      // st_tnsr_i_v(QK_coords[2], QK, acc_reg[2]);
      // st_tnsr_i_v(QK_coords[3], QK, acc_reg[3]);
      // PRINT_REG_VALUE("reg0\n", acc_reg[0]);

      // softmax
      QK_max[0] = v_reduce_max_v_v(QK_max[0]);
      QK_max[1] = v_reduce_max_v_v(QK_max[1]);
      QK_max[2] = v_reduce_max_v_v(QK_max[2]);
      QK_max[3] = v_reduce_max_v_v(QK_max[3]);
      // #ifdef FLOAT32
      printf("QK_max %f\n", QK_max[0][0]);
      printf("QK_max %f\n", QK_max[1][0]);
      printf("QK_max %f\n", QK_max[2][0]);
      printf("QK_max %f\n", QK_max[3][0]);
      // #else
      //     dump_max(QK_max[0]);
      //     dump_max(QK_max[1]);
      //     dump_max(QK_max[2]);
      //     dump_max(QK_max[3]);
      // #endif
      QK_exp_sum[0] = 0.f;
      QK_exp_sum[1] = 0.f;
      QK_exp_sum[2] = 0.f;
      QK_exp_sum[3] = 0.f;

      // printf("cur_kv %d\n", cur_k_head);
      // #ifdef NO_SLM_SOFTMAX
      //       set_coords(QK_coords, cur_k_head);
      //       acc_reg[0] = v_ld_tnsr_i(QK_coords[0], QK);
      //       acc_reg[1] = v_ld_tnsr_i(QK_coords[1], QK);
      //       acc_reg[2] = v_ld_tnsr_i(QK_coords[2], QK);
      //       acc_reg[3] = v_ld_tnsr_i(QK_coords[3], QK);
      // #else
      //       acc_reg[i] = slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)];
      // #endif
      acc_reg[0] = v_sub_v_v(acc_reg[0], QK_max[0]);
      acc_reg[1] = v_sub_v_v(acc_reg[1], QK_max[1]);
      acc_reg[2] = v_sub_v_v(acc_reg[2], QK_max[2]);
      acc_reg[3] = v_sub_v_v(acc_reg[3], QK_max[3]);
#ifdef FLOAT32
      acc_reg[0] = v_exp_f32(acc_reg[0]);
      acc_reg[1] = v_exp_f32(acc_reg[1]);
      acc_reg[2] = v_exp_f32(acc_reg[2]);
      acc_reg[3] = v_exp_f32(acc_reg[3]);
#else
      acc_reg[0] = bf16_exp(acc_reg[0]);
      acc_reg[1] = bf16_exp(acc_reg[1]);
      acc_reg[2] = bf16_exp(acc_reg[2]);
      acc_reg[3] = bf16_exp(acc_reg[3]);
#endif
#ifdef NO_SLM_SOFTMAX

      // set_coords(QK_coords, cur_k_head);
      // st_tnsr_i_v(QK_coords[0], QK, acc_reg[0]);
      // st_tnsr_i_v(QK_coords[1], QK, acc_reg[1]);
      // st_tnsr_i_v(QK_coords[2], QK, acc_reg[2]);
      // st_tnsr_i_v(QK_coords[3], QK, acc_reg[3]);
#else
      slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)] = acc_reg[i];
#endif
      QK_exp_sum[0] = v_add_v_v(QK_exp_sum[0], acc_reg[0]);
      QK_exp_sum[1] = v_add_v_v(QK_exp_sum[1], acc_reg[1]);
      QK_exp_sum[2] = v_add_v_v(QK_exp_sum[2], acc_reg[2]);
      QK_exp_sum[3] = v_add_v_v(QK_exp_sum[3], acc_reg[3]);

            QK_exp_sum[0] = v_reduce_add_v_v(QK_exp_sum[0]);
            QK_exp_sum[1] = v_reduce_add_v_v(QK_exp_sum[1]);
            QK_exp_sum[2] = v_reduce_add_v_v(QK_exp_sum[2]);
            QK_exp_sum[3] = v_reduce_add_v_v(QK_exp_sum[3]);
      #ifdef FLOAT32
            QK_exp_sum[0] = v_div_f32(tmp, QK_exp_sum[0]);
            QK_exp_sum[1] = v_div_f32(tmp, QK_exp_sum[1]);
            QK_exp_sum[2] = v_div_f32(tmp, QK_exp_sum[2]);
            QK_exp_sum[3] = v_div_f32(tmp, QK_exp_sum[3]);
      #else
            QK_exp_sum[0] = bf16_div(QK_exp_sum[0], tmp);
            QK_exp_sum[1] = bf16_div(QK_exp_sum[1], tmp);
            QK_exp_sum[2] = bf16_div(QK_exp_sum[2], tmp);
            QK_exp_sum[3] = bf16_div(QK_exp_sum[3], tmp);
      #endif

      // set_coords(QK_coords, cur_k_head);
#ifdef NO_SLM_SOFTMAX
      // acc_reg[0] = v_ld_tnsr_i(QK_coords[0], QK);
      // acc_reg[1] = v_ld_tnsr_i(QK_coords[1], QK);
      // acc_reg[2] = v_ld_tnsr_i(QK_coords[2], QK);
      // acc_reg[3] = v_ld_tnsr_i(QK_coords[3], QK);
#else
      acc_reg[i] = slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)];
#endif
      acc_reg[0] = v_mul_v_v(acc_reg[0], QK_exp_sum[0]);
      // PRINT_REG_VALUE("result\n", acc_reg[0]);
      acc_reg[1] = v_mul_v_v(acc_reg[1], QK_exp_sum[1]);
      acc_reg[2] = v_mul_v_v(acc_reg[2], QK_exp_sum[2]);
      acc_reg[3] = v_mul_v_v(acc_reg[3], QK_exp_sum[3]);
      set_coords(QK_coords, cur_k_head);
      st_tnsr_i_v(QK_coords[0], QK, acc_reg[0]);
      st_tnsr_i_v(QK_coords[1], QK, acc_reg[1]);
      st_tnsr_i_v(QK_coords[2], QK, acc_reg[2]);
      st_tnsr_i_v(QK_coords[3], QK, acc_reg[3]);
      // PRINT_REG_VALUE("reg0\n", acc_reg[0]);
      // printf("================\n");
      // acc_reg[0] = v_ld_tnsr_i(QK_coords[0], QK);
      // PRINT_REG_VALUE("reg0\n", acc_reg[0]);
    }
    // QK *V gemv
    for (int cur_dim = 0; cur_dim < head_dim; cur_dim += VECTOR_SIZE) {
      V_coords[0] = cur_dim;
      Out_coords[0][0] = cur_dim;
      Out_coords[1][0] = cur_dim;
      Out_coords[2][0] = cur_dim;
      Out_coords[3][0] = cur_dim;
      Out_coords[0][1] = 0;
      Out_coords[1][1] = 0;
      Out_coords[2][1] = 0;
      Out_coords[3][1] = 0;
#pragma loop_taken
      for (int i = 0; i < broadcast_unroll; i++) acc_reg[i] = 0.f;
      for (int broadcast_QK_dim = 0; broadcast_QK_dim < kv_seq_len; broadcast_QK_dim++) {
        QK_coords[0][0] = broadcast_QK_dim;
        QK_coords[1][0] = broadcast_QK_dim;
        QK_coords[2][0] = broadcast_QK_dim;
        QK_coords[3][0] = broadcast_QK_dim;
        set_coords(QK_coords, cur_k_head);
        __global__ float* p_QK0 = gen_addr(QK_coords[0], QK);
        __global__ float* p_QK1 = gen_addr(QK_coords[1], QK);
        __global__ float* p_QK2 = gen_addr(QK_coords[2], QK);
        __global__ float* p_QK3 = gen_addr(QK_coords[3], QK);
        broadcast_reg[0] = v_ld_g_a(p_QK0);
        broadcast_reg[1] = v_ld_g_a(p_QK1);
        broadcast_reg[2] = v_ld_g_a(p_QK2);
        broadcast_reg[3] = v_ld_g_a(p_QK3);

        V_coords[1] = broadcast_QK_dim;
        KV_reg = v_ld_tnsr_i(V_coords, V);

        acc_reg[0] = v_mac_v_v(KV_reg, broadcast_reg[0], acc_reg[0]);
        acc_reg[1] = v_mac_v_v(KV_reg, broadcast_reg[1], acc_reg[1]);
        acc_reg[2] = v_mac_v_v(KV_reg, broadcast_reg[2], acc_reg[2]);
        acc_reg[3] = v_mac_v_v(KV_reg, broadcast_reg[3], acc_reg[3]);
      }
      set_coords(Out_coords,cur_k_head);
      st_tnsr_i_v(Out_coords[0], Out, acc_reg[0]);
      st_tnsr_i_v(Out_coords[1], Out, acc_reg[1]);
      st_tnsr_i_v(Out_coords[2], Out, acc_reg[2]);
      st_tnsr_i_v(Out_coords[3], Out, acc_reg[3]);
    }
  }
}