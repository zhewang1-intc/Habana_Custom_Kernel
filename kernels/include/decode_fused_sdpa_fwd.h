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

#define process_64_kv_reg_num 1

#define PRINT_REG_VALUE(NAME, REG)                                             \
  printf(NAME);                                                                \
  for (int i = 0; i < 64; i++)                                                 \
    printf("%f ", REG[i]);                                                     \
  printf("\n");

__local__ float64 slm_test[1024 * 16 / 4 / 64];
void main(tensor Q, tensor K, tensor QK, tensor V, tensor Out) {
  const int5 Q_head_start = get_index_space_offset();
  const int5 Q_head_end = get_index_space_size() + Q_head_start;

  const int depthStep = 64;
  const int head_dim = get_dim_size(Q, 0);
  const int q_seq_len = get_dim_size(Q, 1);
  const int kv_seq_len = get_dim_size(K, 0);
  const int q_head_num = get_dim_size(Q, 2);
  const int kv_head_num = get_dim_size(K, 2);
  const int shared_kv_head = q_head_num / kv_head_num;
  int5 Q_coords = {0};
  int5 K_coords = {0};
  int5 QK_coords = {0};
  //     int5 V_coords = {0};
  //     int5 Out_coords = {0};

  float64 sqrt_dk = 2.f; // todo: make it as param.

  float64 broadcast_Q_reg;
  float64 K_reg[process_64_kv_reg_num];
  float64 QK_acc[process_64_kv_reg_num];
  float64 QK_max = -9999999999.f;
  float64 QK_exp_sum;
  float64 tmp = 1.f;

  for (int cur_q_head = Q_head_start[0]; cur_q_head < Q_head_end[0];
       cur_q_head++) {
    Q_coords[2] = cur_q_head;
    K_coords[2] = cur_q_head / shared_kv_head;
    QK_coords[2] = cur_q_head;
    printf("q_head: %d\n", cur_q_head);
    printf("kv_head: %d\n", K_coords[2]);
    // gemv
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len;
         cur_kv_seq += depthStep * process_64_kv_reg_num) {
      for (int i = 0; i < process_64_kv_reg_num; i++)
        QK_acc[i] = 0.f;
      for (int broadcast_Q_dim = 0; broadcast_Q_dim < head_dim;
           broadcast_Q_dim++) {
        // printf("cur_head_dim: %d\n", broadcast_Q_dim);
        Q_coords[0] = broadcast_Q_dim;
        K_coords[1] = broadcast_Q_dim;
        __global__ float *p_Q = gen_addr(Q_coords, Q);
        broadcast_Q_reg = v_f32_ld_g(p_Q);
        // PRINT_REG_VALUE("broadcast_Q_reg\n", broadcast_Q_reg);
        for (int i = 0; i < process_64_kv_reg_num; i++) {
          K_coords[0] = cur_kv_seq + i * 64;
          // printf("cur_kv_seq: %d\n", K_coords[0]);
          K_reg[i] = v_f32_ld_tnsr_b(K_coords, K);
          // PRINT_REG_VALUE("K_reg\n", K_reg[i]);
          QK_acc[i] = v_f32_mac_b(K_reg[i], broadcast_Q_reg, QK_acc[i]);
          // PRINT_REG_VALUE("QK_acc_reg\n", QK_acc[i]);
        }
      }
      for (int i = 0; i < process_64_kv_reg_num; i++) {
        QK_coords[0] = cur_kv_seq + i * 64;
        v_f32_st_tnsr(
            QK_coords, QK,
            QK_acc[i]); // TODO(zhe): move to slm when kv_seq is small.
        // slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)] = QK_acc[i];
        QK_max = v_f32_max_b(QK_acc[i], QK_max);
      }
    }
    // softmax
    QK_max = v_f32_reduce_max(QK_max);
    printf("Q_max: %f\n", QK_max[0]);
    QK_exp_sum = 0.f;
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len;
         cur_kv_seq += depthStep * process_64_kv_reg_num) {
      for (int i = 0; i < process_64_kv_reg_num; i++) {
        QK_coords[0] = cur_kv_seq + i * 64;
        QK_acc[i] = v_f32_ld_tnsr_b(QK_coords, QK);
        // QK_acc[i] = slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)];
        // PRINT_REG_VALUE("QK\n", QK_acc[i]);
        QK_acc[i] = v_f32_sub_b(QK_acc[i], QK_max);
        // PRINT_REG_VALUE("sub exp\n", QK_acc[i]);
        QK_acc[i] = v_exp_f32(QK_acc[i]);
        v_f32_st_tnsr(QK_coords, QK, QK_acc[i]);
        // slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)] = QK_acc[i];
        QK_exp_sum = v_f32_add_b(QK_exp_sum, QK_acc[i]);
      }
    }
    QK_exp_sum = v_f32_reduce_add(QK_exp_sum);
    // PRINT_REG_VALUE("exp_sum\n", QK_exp_sum);
    QK_exp_sum = v_div_f32(tmp, QK_exp_sum);
    // PRINT_REG_VALUE("div_exp_sum\n", QK_exp_sum);
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len;
         cur_kv_seq += depthStep * process_64_kv_reg_num) {
      for (int i = 0; i < process_64_kv_reg_num; i++) {
        QK_coords[0] = cur_kv_seq + i * 64;
        QK_acc[i] = v_f32_ld_tnsr_b(QK_coords, QK);
        // QK_acc[i] =
            // v_f32_mul_b(slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)],
                        // QK_exp_sum);
        QK_acc[i] = v_f32_mul_b(QK_acc[i], QK_exp_sum);
        v_f32_st_tnsr(QK_coords, QK, QK_acc[i]);
      }
    }
  }
}
