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

#define process_64_kv_reg_num 2
#define NO_SLM_SOFTMAX

#define PRINT_REG_VALUE(NAME, REG) \
  printf(NAME);                    \
  for (int i = 0; i < 64; i++)     \
    printf("%f ", REG[i]);         \
  printf("\n");

__local__ float64 slm_test[1024 * 16 / 4 / 64];
void main(tensor Q, tensor K, tensor QK, tensor V, tensor Out)
{
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
  int5 V_coords = {0};
  int5 Out_coords = {0};

  float64 sqrt_dk = 2.f; // todo: make it as param.

  float64 broadcast_reg;
  float64 KV_reg[process_64_kv_reg_num];
  float64 acc_reg[process_64_kv_reg_num];
  float64 QK_max = -9999999999.f;
  float64 QK_exp_sum;
  float64 tmp = 1.f;

  for (int cur_q_head = Q_head_start[0]; cur_q_head < Q_head_end[0];
       cur_q_head++)
  {
    Q_coords[2] = cur_q_head;
    K_coords[2] = cur_q_head / shared_kv_head;
    V_coords[2] = cur_q_head / shared_kv_head;
    QK_coords[2] = cur_q_head;
    Out_coords[2] = cur_q_head;
    // gemv
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len;
         cur_kv_seq += depthStep * process_64_kv_reg_num)
    {
#pragma loop_taken
      for (int i = 0; i < process_64_kv_reg_num; i++)
        acc_reg[i] = 0.f;
      for (int broadcast_Q_dim = 0; broadcast_Q_dim < head_dim;
           broadcast_Q_dim++)
      {
        Q_coords[0] = broadcast_Q_dim;
        K_coords[1] = broadcast_Q_dim;
        __global__ float *p_Q = gen_addr(Q_coords, Q);
        broadcast_reg = v_f32_ld_g(p_Q);
#pragma loop_taken
        for (int i = 0; i < process_64_kv_reg_num; i++)
        {
          K_coords[0] = cur_kv_seq + i * 64;
          KV_reg[i] = v_f32_ld_tnsr_b(K_coords, K);
          acc_reg[i] = v_f32_mac_b(KV_reg[i], broadcast_reg, acc_reg[i]);
        }
      }
#pragma loop_taken
      for (int i = 0; i < process_64_kv_reg_num; i++)
      {
        QK_coords[0] = cur_kv_seq + i * 64;
        QK_max = v_f32_max_b(acc_reg[i], QK_max);
#ifdef NO_SLM_SOFTMAX
        v_f32_st_tnsr(
            QK_coords, QK,
            acc_reg[i]); // TODO(zhe): move to slm when kv_seq is small.
#else
        slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)] = acc_reg[i];
#endif
      }
    }
    // softmax
    QK_max = v_f32_reduce_max(QK_max);
    QK_exp_sum = 0.f;
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len;
         cur_kv_seq += depthStep * process_64_kv_reg_num)
    {
#pragma loop_taken
      for (int i = 0; i < process_64_kv_reg_num; i++)
      {
#ifdef NO_SLM_SOFTMAX
        QK_coords[0] = cur_kv_seq + i * 64;
        acc_reg[i] = v_f32_ld_tnsr_b(QK_coords, QK);
#else
        acc_reg[i] = slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)];
#endif
        acc_reg[i] = v_f32_sub_b(acc_reg[i], QK_max);
        acc_reg[i] = v_exp_f32(acc_reg[i]);
#ifdef NO_SLM_SOFTMAX
        v_f32_st_tnsr(QK_coords, QK, acc_reg[i]);
#else
        slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)] = acc_reg[i];
#endif
        QK_exp_sum = v_f32_add_b(QK_exp_sum, acc_reg[i]);
      }
    }
    QK_exp_sum = v_f32_reduce_add(QK_exp_sum);
    QK_exp_sum = v_div_f32(tmp, QK_exp_sum);
    for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len;
         cur_kv_seq += depthStep * process_64_kv_reg_num)
    {
#pragma loop_taken
      for (int i = 0; i < process_64_kv_reg_num; i++)
      {
        QK_coords[0] = cur_kv_seq + i * 64;
#ifdef NO_SLM_SOFTMAX
        acc_reg[i] = v_f32_ld_tnsr_b(QK_coords, QK);
#else
        acc_reg[i] = slm_test[i + cur_kv_seq / (64 * process_64_kv_reg_num)];
#endif
        acc_reg[i] = v_f32_mul_b(acc_reg[i], QK_exp_sum);
        v_f32_st_tnsr(QK_coords, QK, acc_reg[i]);
      }
    }
    // QK * V gemv
    for (int cur_dim = 0; cur_dim < head_dim;
         cur_dim += depthStep * process_64_kv_reg_num)
    {
      V_coords[0] = cur_dim;
      Out_coords[0] = cur_dim;
#pragma loop_taken
      for (int i = 0; i < process_64_kv_reg_num; i++)
        acc_reg[i] = 0.f;
      for (int broadcast_QK_dim = 0; broadcast_QK_dim < kv_seq_len;
           broadcast_QK_dim++)
      {
        QK_coords[0] = broadcast_QK_dim;
        V_coords[1] = broadcast_QK_dim;
        __global__ float *p_QK = gen_addr(QK_coords, QK);
        broadcast_reg = v_f32_ld_g(p_QK);
#pragma loop_taken
        for (int i = 0; i < process_64_kv_reg_num; i++)
        {
          KV_reg[i] = v_f32_ld_tnsr_b(V_coords, V);
          acc_reg[i] = v_f32_mac_b(KV_reg[i], broadcast_reg, acc_reg[i]);
        }
#pragma loop_taken
        for (int i = 0; i < process_64_kv_reg_num; i++)
          v_f32_st_tnsr(Out_coords, Out, acc_reg[i]);
      }
    }
  }
}