/**********************************************************************
Copyright (c) 2022 Habana Labs.

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

#ifndef DECODE_FUSED_SDPA_TEST_HPP
#define DECODE_FUSED_SDPA_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "decode_fused_sdpa.hpp"
#include "entry_points.hpp"

template <typename T>
class DecodeFusedSdpaTest : public TestBase
{
public:
    DecodeFusedSdpaTest() {}
    ~DecodeFusedSdpaTest() {}
    int runTest();

    static void deocde_fused_sdpa_ref(
        test::Tensor<T, 3> &Q,
        test::Tensor<T, 3> &K,
        test::Tensor<T, 3> &QK,
        test::Tensor<T, 3> &V,
        test::Tensor<T, 3> &Out);

private:
    DecodeFusedSdpaTest(const DecodeFusedSdpaTest &other) = delete;
    DecodeFusedSdpaTest &operator=(const DecodeFusedSdpaTest &other) = delete;
};

template <typename T>
void DecodeFusedSdpaTest<T>::deocde_fused_sdpa_ref(
    test::Tensor<T, 3> &Q,
    test::Tensor<T, 3> &K,
    test::Tensor<T, 3> &QK,
    test::Tensor<T, 3> &V,
    test::Tensor<T, 3> &Out)
{
    int q_head = Q.Size(2);
    int kv_head = K.Size(2);
    int head_dim = Q.Size(0);
    int kv_seq = K.Size(0);
    int q_seq = Q.Size(1);
    int kv_seq_len = K.Size(0);
    int shared_kv_head = q_head / kv_head;
    int Q_coords[3] = {0};
    int K_coords[3] = {0};
    int QK_coords[3] = {0};
    int V_coords[3] = {0};
    int Out_coords[3] = {0};
    T QK_max = -9999999.f;
    bool is_bf16 = true;
    if constexpr (std::is_same<T, float>::value)
        is_bf16 = false;
    for (int cur_q_head = 0; cur_q_head < q_head; cur_q_head++)
    {
        T exp_sum = 0.f;
        Q_coords[2] = cur_q_head;
        K_coords[2] = cur_q_head / shared_kv_head;
        V_coords[2] = cur_q_head / shared_kv_head;
        QK_coords[2] = cur_q_head;
        Out_coords[2] = cur_q_head;
        for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq++)
        {
            QK_coords[0] = cur_kv_seq;
            QK.SetElement(QK_coords, 0.f);
        }
        for (int cur_dim = 0; cur_dim < head_dim; cur_dim++)
        {
            Q_coords[0] = cur_dim;
            K_coords[1] = cur_dim;
            Out_coords[0] = cur_dim;
            Out.SetElement(Out_coords, 0.f);
            for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq++)
            {
                K_coords[0] = cur_kv_seq;
                QK_coords[0] = cur_kv_seq;
                QK.SetElement(QK_coords, QK.ElementAt(QK_coords) + Q.ElementAt(Q_coords) * K.ElementAt(K_coords));
            }
        }
        for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq++)
        {
            QK_coords[0] = cur_kv_seq;
            QK_max = QK_max > QK.ElementAt(QK_coords) ? QK_max : QK.ElementAt(QK_coords);
        }
        std::cout << "ref QK_max: " << float(QK_max) << std::endl;
        for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq++)
        {
            QK_coords[0] = cur_kv_seq;
            QK.SetElement(QK_coords, QK.ElementAt(QK_coords) - QK_max);
            QK.SetElement(QK_coords, expf(float(QK.ElementAt(QK_coords))));
            exp_sum = exp_sum + QK.ElementAt(QK_coords);
        }
        for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq++)
        {
            QK_coords[0] = cur_kv_seq;
            QK.SetElement(QK_coords, QK.ElementAt(QK_coords) / exp_sum);
        }
        for (int cur_kv_seq = 0; cur_kv_seq < kv_seq_len; cur_kv_seq++)
        {
            QK_coords[0] = cur_kv_seq;
            V_coords[1] = cur_kv_seq;
            for (int cur_dim = 0; cur_dim < head_dim; cur_dim++)
            {
                V_coords[0] = cur_dim;
                Out_coords[0] = cur_dim;
                Out.SetElement(Out_coords, Out.ElementAt(Out_coords) + QK.ElementAt(QK_coords) * V.ElementAt(V_coords));
            }
        }
    }
}

template <typename T>
int DecodeFusedSdpaTest<T>::runTest()
{

    // Initalize input size
    const int q_head = 32;
    const int q_seq = 1;
    const int head_dim = 128;
    const int kv_head = 8;
    const int kv_seq = 1024;

    // Initalize inputs
    uint64_t q_init[] = {head_dim, 1, q_head};
    uint64_t tmp_init[] = {kv_seq, 3, q_head};
    uint64_t k_init[] = {kv_seq, head_dim, kv_head};
    uint64_t v_init[] = {head_dim, kv_seq, kv_head};
    test::Tensor<T, 3> Q(q_init);
    test::Tensor<T, 3> K(k_init);
    test::Tensor<T, 3> QK(tmp_init);
    test::Tensor<T, 3> QK_ref(tmp_init);
    test::Tensor<T, 3> V(v_init);
    test::Tensor<T, 3> Out(q_init);
    test::Tensor<T, 3> Out_ref(q_init);
    Q.FillWithData(0);
    K.FillWithData(1);
    V.FillWithData(1);
    // execute reference implementation of the kernel.
    this->deocde_fused_sdpa_ref(Q, K, QK_ref, V, Out_ref);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;
    // m_in_defs.nodeParams.nodeParams = &def;
    m_in_defs.inputTensorNr = 4;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), Q);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), K);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), QK);
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]), V);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), Out);

    tpc_lib_api::GuidInfo *guids = nullptr;
    unsigned kernelCount = 0;
    tpc_lib_api::GlueCodeReturn result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    guids = new tpc_lib_api::GuidInfo[kernelCount];
    result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    if constexpr (std::is_same_v<T, float>)
        strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_DECODE_FUSED_SDPA_F32].name);
    else
        strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_DECODE_FUSED_SDPA_BF16].name);
    result = InstantiateTpcKernel(&m_in_defs, &m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(Q.GetTensorDescriptor());
    vec.push_back(K.GetTensorDescriptor());
    vec.push_back(QK.GetTensorDescriptor());
    vec.push_back(V.GetTensorDescriptor());
    vec.push_back(Out.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);
    for (int element = 0; element < Out_ref.ElementCount(); element++)
    {
        if (abs(float(Out.Data()[element]) - float(Out_ref.Data()[element])) > 10e-3)
        {
            std::cout << "err idx:" << element << ", value: " << float(Out.Data()[element]) << " vs " << float(Out_ref.Data()[element]) << std::endl;
            std::cout << "decode sdpa test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "decode sdpa test pass!!" << std::endl;

    return 0;
}
#endif /* CAST_F16_TO_I16_TEST_HPP */
