/**********************************************************************
Copyright (c) 2022 Habana Labs.

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

#ifndef MERGE_EXPERT_TEST_HPP
#define MERGE_EXPERT_TEST_HPP

#include "entry_points.hpp"
#include "merge_expert.hpp"
#include "tensor.h"
#include "test_base.hpp"

template <typename T>
class MergeExpertTest : public TestBase {
 public:
  MergeExpertTest() {}
  ~MergeExpertTest() {}
  int runTest();

  static void merge_expert_ref(test::Tensor<T, 3> &input,
                               test::Tensor<int, 2> &indices,
                               test::Tensor<T, 2> &weights,
                               test::Tensor<T, 2> &output);

 private:
  MergeExpertTest(const MergeExpertTest &other) = delete;
  MergeExpertTest &operator=(const MergeExpertTest &other) = delete;
};

template <typename T>
void assign_info(test::Tensor<int, 2> &indices, test::Tensor<T, 2> &weights) {
  int tokens = indices.Size(1);
  int expert_num = indices.Size(0);
  int coords[2] = {0};
  for (int i = 0; i < tokens; i++) {
    coords[1] = i;
    float sum = 0;
    for (int j = 0; j < expert_num; j++) {
      coords[0] = j;
      indices.SetElement(coords, rand() % expert_num);
      weights.SetElement(coords,
                         static_cast<T>(rand()) / static_cast<float>(RAND_MAX));
      sum += weights.ElementAt(coords);
    }
    for (int j = 0; j < expert_num; j++) {
      coords[0] = j;
      weights.SetElement(coords, weights.ElementAt(coords) / sum);
    }
  }
}

template <typename T>
void MergeExpertTest<T>::merge_expert_ref(test::Tensor<T, 3> &input,
                                          test::Tensor<int, 2> &indices,
                                          test::Tensor<T, 2> &weights,
                                          test::Tensor<T, 2> &output) {
  int token_num = input.Size(1);
  int total_expert_num = input.Size(2);
  int dim = input.Size(0);
  int select_expert_num = indices.Size(1);
  int input_coords[3] = {0};
  int output_coords[2] = {0};
  int info_coords[2] = {0};
  for (int token_id = 0; token_id < token_num; token_id++) {
    input_coords[1] = token_id;
    output_coords[1] = token_id;
    info_coords[1] = token_id;
    for (int expert_iter = 0; expert_iter < select_expert_num; expert_iter++) {
      info_coords[0] = expert_iter;
      int expert_id = indices.ElementAt(info_coords);
      input_coords[2] = expert_id;
      T weight = weights.ElementAt(info_coords);
      for (int dim_id = 0; dim_id < dim; dim_id++) {
        input_coords[0] = dim_id;
        output_coords[0] = dim_id;
        output.SetElement(output_coords,
                          output.ElementAt(output_coords) +
                              input.ElementAt(input_coords) * weight);
      }
    }
  }
}

template <typename T>
int MergeExpertTest<T>::runTest() {
  // Initalize input size
  const int token = 8;
  const int dim = 64;
  const int select_expert_num = 4;
  const int total_expert_num = 8;
  // Initalize inputs
  uint64_t in_init[] = {dim, token, total_expert_num};
  uint64_t out_init[] = {dim, token};
  uint64_t info_init[] = {select_expert_num, token};
  test::Tensor<T, 3> input(in_init);
  test::Tensor<int, 2> indices(info_init);
  test::Tensor<T, 2> weights(info_init);
  test::Tensor<T, 2> out(out_init);
  test::Tensor<T, 2> out_ref(out_init);
  input.FillWithData(0);
  out.FillWithData();
  out_ref.FillWithData();
  assign_info(indices, weights);

  this->merge_expert_ref(input, indices, weights, out_ref);

  // generate input for query call
  m_in_defs.deviceId = tpc_lib_api::DEVICE_ID_GAUDI2;

  m_in_defs.inputTensorNr = 3;
  LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
  LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), indices);
  LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), weights);

  m_in_defs.outputTensorNr = 1;
  LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), out);

  tpc_lib_api::GuidInfo *guids = nullptr;
  unsigned kernelCount = 0;
  tpc_lib_api::GlueCodeReturn result =
      GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
  guids = new tpc_lib_api::GuidInfo[kernelCount];
  result = GetKernelGuids(tpc_lib_api::DEVICE_ID_GAUDI2, &kernelCount, guids);
  if (result != tpc_lib_api::GLUE_SUCCESS) {
    std::cout << "Can't get kernel name!! " << result << std::endl;
    ReleaseKernelNames(guids, kernelCount);
    return -1;
  }

  if constexpr (std::is_same_v<T, float>)
    strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_MERGE_EXPERT_F32].name);
  else
    strcpy(m_in_defs.guid.name, guids[GAUDI2_KERNEL_MERGE_EXPERT_BF16].name);
  result = InstantiateTpcKernel(&m_in_defs, &m_out_defs);
  if (result != tpc_lib_api::GLUE_SUCCESS) {
    std::cout << "Glue test failed, can't load kernel " << result << std::endl;
    ReleaseKernelNames(guids, kernelCount);
    return -1;
  }

  // generate and load tensor descriptors
  std::vector<TensorDesc2> vec;
  vec.push_back(input.GetTensorDescriptor());
  vec.push_back(indices.GetTensorDescriptor());
  vec.push_back(weights.GetTensorDescriptor());
  vec.push_back(out.GetTensorDescriptor());

  // execute a simulation of the kernel using TPC simulator,
  TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
  ReleaseKernelNames(guids, kernelCount);
  for (int element = 0; element < out_ref.ElementCount(); element++) {
    if (abs(float(out.Data()[element]) - float(out_ref.Data()[element])) >
        10e-3) {
      std::cout << "err idx:" << element
                << ", value: " << float(out.Data()[element]) << " vs "
                << float(out_ref.Data()[element]) << std::endl;
      std::cout << "merge expert test failed!!" << std::endl;
      return -1;
    }
  }
  std::cout << "merge expert test pass!!" << std::endl;

  return 0;
}
#endif /* CAST_F16_TO_I16_TEST_HPP */
