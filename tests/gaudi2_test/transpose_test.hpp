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

#ifndef TRANSPOSE_TEST_HPP
#define TRANSPOSE_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "transpose.hpp"
#include "entry_points.hpp"

template <typename T>
class TransposeTest : public TestBase
{
public:
    TransposeTest() {}
    ~TransposeTest() {}
    int runTest();

    static void transpose_ref(
        test::Tensor<T, 4> &a,
        test::Tensor<T, 4> &b,
        int axes[4]);

private:
    TransposeTest(const TransposeTest &other) = delete;
    TransposeTest &operator=(const TransposeTest &other) = delete;
};

template <typename T>
void TransposeTest<T>::transpose_ref(test::Tensor<T, 4> &ifm,
                                     test::Tensor<T, 4> &ofm,
                                     int axes[4])
{
    int depthEnd = ifm.Size(0);
    int widthEnd = ifm.Size(1);
    int heightEnd = ifm.Size(2);
    int batchEnd = ifm.Size(3);

    int coordsIn[4]{0};
    int coordsOut[4]{0};

    for (int d = 0; d < depthEnd; d++)
    {
        coordsIn[0] = d;
        coordsOut[axes[0]] = d;

        for (int b = 0; b < batchEnd; b++)
        {
            coordsIn[3] = b;
            coordsOut[axes[3]] = b;

            for (int h = 0; h < heightEnd; h++)
            {
                coordsIn[2] = h;
                coordsOut[axes[2]] = h;

                for (int w = 0; w < widthEnd; w++)
                {
                    coordsIn[1] = w;
                    coordsOut[axes[1]] = w;
                    ofm.SetElement(coordsOut, ifm.ElementAt(coordsIn));
                }
            }
        }
    }
}

template <typename T>
int TransposeTest<T>::runTest()
{

    // Initalize input size
    const int row = 1;
    const int col = 64;
    const int n_contex = 128;
    // Initalize inputs
    uint64_t a_init[] = {col, row, 1, 1};
    uint64_t b_init[] = {row, col, 1, 1};
    test::Tensor<T, 4> a(a_init);
    test::Tensor<T, 4> b(b_init);
    test::Tensor<T, 4> b_ref(b_init);
    a.FillWithData(0);
    TransposeGaudi2::TpcTransParams p;
    p.axes[0] = 1;
    p.axes[1] = 0;
    p.axes[2] = 2;
    p.axes[3] = 3;
    m_in_defs.nodeParams.nodeParams = &p;
    this->transpose_ref(a, b_ref, p.axes);

    // generate input for query call
    m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;

    m_in_defs.inputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), a);

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), b);

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
        strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_TRANSPOSE_F32].name);
    else
        strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_TRANSPOSE_BF16].name);
    result = InstantiateTpcKernel(&m_in_defs, &m_out_defs);
    if (result != tpc_lib_api::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(guids, kernelCount);
        return -1;
    }

    // generate and load tensor descriptors
    std::vector<TensorDesc2> vec;
    vec.push_back(a.GetTensorDescriptor());
    vec.push_back(b.GetTensorDescriptor());

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(guids, kernelCount);
    for (int element = 0; element < b_ref.ElementCount(); element++)
    {
        if (abs(float(b.Data()[element]) - float(b_ref.Data()[element])) > 10e-3)
        {
            std::cout << "err idx:" << element << ", value: " << float(b.Data()[element]) << " vs " << float(b_ref.Data()[element]) << std::endl;
            std::cout << "decode sdpa test failed!!" << std::endl;
            return -1;
        }
    }
    std::cout << "decode sdpa test pass!!" << std::endl;

    return 0;
}
#endif /* CAST_F16_TO_I16_TEST_HPP */
