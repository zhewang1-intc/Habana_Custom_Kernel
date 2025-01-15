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

#define UNROLL_KV 1 // should same as repeat_kv_num.
#define REPEAT_KV 4
#define UNROLL_Q UNROLL_KV *REPEAT_KV

#define PRINT_REG_VALUE(NAME, REG) \
  printf(NAME);                    \
  for (int i = 0; i < 64; i++)     \
  {                                \
    printf("idx: %d  ", i);        \
    printf("%f\n", REG[i]);        \
  }

void dump_max(bfloat128 reg)
{
  float128 t = convert_bfloat128_to_float128(reg, SW_RHNE);
  printf("QK_max %f\n", t.v1[0]);
}

#ifdef FLOAT32
VECTOR mul_sqrt_dk_exp(VECTOR reg, float64 sqrt_dk)
{
  reg = v_mul_v_v(reg, sqrt_dk);
  reg = v_exp_f32(reg);
  return reg;
}
#else
VECTOR mul_sqrt_dk_exp(VECTOR reg, float64 sqrt_dk)
{
  float128 tmp_128 = convert_bfloat128_to_float128(reg, SW_RHNE);
  tmp_128.v1 = v_f32_mul_b(tmp_128.v1, sqrt_dk);
  tmp_128.v2 = v_f32_mul_b(tmp_128.v2, sqrt_dk);
  tmp_128.v1 = v_exp_f32(tmp_128.v1);
  tmp_128.v2 = v_exp_f32(tmp_128.v2);
  reg = convert_float128_to_bfloat128(tmp_128, SW_RHNE);
  return reg;
}
#endif

#define aso_init() \
  int count = 0;   \
  set_semaphore_value(0);

#define aso_wait()                          \
  {                                         \
    aso(SW_INC | SW_VPU);                   \
    volatile int a = get_semaphore_value(); \
    while (a == count)                      \
    {                                       \
      a = get_semaphore_value();            \
    }                                       \
    count++;                                \
  }

#ifdef FLOAT32
VECTOR v_div(VECTOR reg, float64 tmp)
{
  reg = v_div_f32(tmp, reg);
  return reg;
}
#else
VECTOR v_div(VECTOR reg, float64 tmp)
{
  float128 tmp_128 = convert_bfloat128_to_float128(reg, SW_RHNE);
  tmp_128.v1 = v_div_f32(tmp, tmp_128.v1);
  tmp_128.v2 = v_div_f32(tmp, tmp_128.v2);
  reg = convert_float128_to_bfloat128(tmp_128, SW_RHNE);
  return reg;
}
#endif

__local__ VECTOR bk_reg[1024 * REPEAT_KV / VECTOR_SIZE];
void main(tensor Q, tensor K, tensor QK, tensor V, tensor kv_seq_len_ts, tensor Out, float sqrt_dk)
{
  const int5 batch_start = get_index_space_offset();
  const int5 batch_end = get_index_space_size() + batch_start;

  FpIntUnion minusInf;
#ifdef FLOAT32
  minusInf.i = 0xff800000;
#else
  minusInf.i = 0xff80;
#endif

  aso_init();

  const int head_dim = get_dim_size(Q, 0);
  // const int kv_seq_len = get_dim_size(K, 0);
  const int kv_head_num = get_dim_size(K, 2);

  int5 Q_coords[UNROLL_Q];
  int5 K_coords = {0};
  int5 QK_coords[UNROLL_Q];
  int5 V_coords = {0};
  int5 Out_coords[UNROLL_Q];
  int5 kv_seq_len_coords = {0};

  float64 sqrt_dk_reg = sqrt_dk;
  VECTOR minus_inf_reg = minusInf.f;

  VECTOR broadcast_reg[UNROLL_Q];
  VECTOR KV_regs[UNROLL_KV];
  VECTOR acc_reg[UNROLL_Q];
  VECTOR QK_max[UNROLL_Q];
  VECTOR QK_exp_sum[UNROLL_Q];
  // VECTOR bk_reg[1024 * REPEAT_KV / VECTOR_SIZE];
  float64 tmp = 1.f;

#pragma unroll(UNROLL_Q)
  for (int i = 0; i < UNROLL_Q; i++)
  {
    QK_coords[i][1] = 0;
    Out_coords[i][1] = 0;
    Q_coords[i][1] = 0;
  }

  for (int cur_batch = batch_start[0]; cur_batch < batch_end[0]; cur_batch++)
  {
#pragma unroll(UNROLL_Q)
    for (int i = 0; i < UNROLL_Q; i++)
    {
      QK_coords[i][3] = cur_batch;
      Out_coords[i][3] = cur_batch;
      Q_coords[i][3] = cur_batch;
    }
    K_coords[3] = cur_batch;
    V_coords[3] = cur_batch;
    kv_seq_len_coords[0] = cur_batch;
    __global__ int *cur_batch_kv_seq_len_ptr = gen_addr(kv_seq_len_coords, kv_seq_len_ts);
    int kv_seq_len = s_i32_ld_g(cur_batch_kv_seq_len_ptr);
    int pad_kv_seq_len = (kv_seq_len + VECTOR_SIZE - 1) / VECTOR_SIZE;
    for (int cur_kv_head = 0; cur_kv_head < kv_head_num; cur_kv_head += UNROLL_KV)
    {
#pragma unroll(UNROLL_Q)
      for (int i = 0; i < UNROLL_Q; i++)
      {
        QK_coords[i][2] = cur_kv_head * REPEAT_KV + i;
        Out_coords[i][2] = cur_kv_head * REPEAT_KV + i;
        Q_coords[i][2] = cur_kv_head * REPEAT_KV + i;
      }
#pragma unroll(UNROLL_Q)
      for (int i = 0; i < UNROLL_Q; i++)
        QK_max[i] = 0.f;
      for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq += VECTOR_SIZE)
      {
        bool256 pred = bv_u_cmp_geq_v_s(cur_kv_seq + V_LANE_ID, kv_seq_len);
        K_coords[0] = cur_kv_seq;
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = 0.f;

        for (int broadcast_Q_dim = 0; broadcast_Q_dim < head_dim; broadcast_Q_dim++)
        {
#pragma unroll(UNROLL_Q)
          for (int i = 0; i < UNROLL_Q; i++)
            Q_coords[i][0] = broadcast_Q_dim;

#pragma unroll(UNROLL_Q)
          for (int i = 0; i < UNROLL_Q; i++)
            broadcast_reg[i] = v_ld_g_a(gen_addr(Q_coords[i], Q));
          K_coords[1] = broadcast_Q_dim;
#pragma unroll(UNROLL_KV)
          for (int i = 0; i < UNROLL_KV; i++)
          {
            K_coords[2] = cur_kv_head + i;
            KV_regs[i] = v_ld_tnsr_i(K_coords, K);
          }
#pragma unroll(UNROLL_Q)
          for (int i = 0; i < UNROLL_Q; i++)
            acc_reg[i] = v_mac_v_v(KV_regs[i / REPEAT_KV], broadcast_reg[i], acc_reg[i]);
        }

#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = v_mov_v_vb(minus_inf_reg, acc_reg[i], pred, 0);

#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          QK_max[i] = v_max_v_v(acc_reg[i], QK_max[i]);

          // #pragma unroll(UNROLL_Q)
          //         for (int i = 0; i < UNROLL_Q; i++) QK_coords[i][0] = cur_kv_seq;

#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          bk_reg[cur_kv_seq / VECTOR_SIZE + i * pad_kv_seq_len] = acc_reg[i];
        //  st_tnsr_i_v(QK_coords[i], QK, acc_reg[i]);
      }

#pragma unroll(UNROLL_Q)
      for (int i = 0; i < UNROLL_Q; i++)
        QK_max[i] = v_reduce_max_v_v(QK_max[i]);

#pragma unroll(UNROLL_Q)
      for (int i = 0; i < UNROLL_Q; i++)
        QK_exp_sum[i] = 0.f;

      // aso_wait();
      for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq += VECTOR_SIZE)
      {
// #pragma unroll(UNROLL_Q)
//         for (int i = 0; i < UNROLL_Q; i++) QK_coords[i][0] = cur_kv_seq;
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = bk_reg[cur_kv_seq / VECTOR_SIZE + i * pad_kv_seq_len];
          //  acc_reg[i] = v_ld_tnsr_i(QK_coords[i], QK);
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = v_sub_v_v(acc_reg[i], QK_max[i]);
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = mul_sqrt_dk_exp(acc_reg[i], sqrt_dk_reg);
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          bk_reg[cur_kv_seq / VECTOR_SIZE + i * pad_kv_seq_len] = acc_reg[i];
          // st_tnsr_i_v(QK_coords[i], QK, acc_reg[i]);
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          QK_exp_sum[i] = v_add_v_v(QK_exp_sum[i], acc_reg[i]);
      }
#pragma unroll(UNROLL_Q)
      for (int i = 0; i < UNROLL_Q; i++)
        QK_exp_sum[i] = v_reduce_add_v_v(QK_exp_sum[i]);

#pragma unroll(UNROLL_Q)
      for (int i = 0; i < UNROLL_Q; i++)
        QK_exp_sum[i] = v_div(QK_exp_sum[i], tmp);

      // aso_wait();
      for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq += VECTOR_SIZE)
      {
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          QK_coords[i][0] = cur_kv_seq;
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = bk_reg[cur_kv_seq / VECTOR_SIZE + i * pad_kv_seq_len];
          //  acc_reg[i] = v_ld_tnsr_i(QK_coords[i], QK);
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = v_mul_v_v(acc_reg[i], QK_exp_sum[i]);
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          st_tnsr_i_v(QK_coords[i], QK, acc_reg[i]);
      }

      aso_wait();

      for (int cur_dim = 0; cur_dim < head_dim; cur_dim += VECTOR_SIZE)
      {
        V_coords[0] = cur_dim;
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          Out_coords[i][0] = cur_dim;
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          acc_reg[i] = 0.f;
        for (int broadcast_QK_dim = 0; broadcast_QK_dim < kv_seq_len; broadcast_QK_dim++)
        {
#pragma unroll(UNROLL_Q)
          for (int i = 0; i < UNROLL_Q; i++)
            QK_coords[i][0] = broadcast_QK_dim;
          V_coords[1] = broadcast_QK_dim;
#pragma unroll(UNROLL_Q)
          for (int i = 0; i < UNROLL_Q; i++)
            broadcast_reg[i] = v_ld_g_a(gen_addr(QK_coords[i], QK));
#pragma unroll(UNROLL_KV)
          for (int i = 0; i < UNROLL_KV; i++)
          {
            V_coords[2] = cur_kv_head + i;
            KV_regs[i] = v_ld_tnsr_i(V_coords, V);
          }
#pragma unroll(UNROLL_Q)
          for (int i = 0; i < UNROLL_Q; i++)
            acc_reg[i] = v_mac_v_v(KV_regs[i / REPEAT_KV], broadcast_reg[i], acc_reg[i]);
        }
#pragma unroll(UNROLL_Q)
        for (int i = 0; i < UNROLL_Q; i++)
          st_tnsr_i_v(Out_coords[i], Out, acc_reg[i]);
      }
    }
  }
}