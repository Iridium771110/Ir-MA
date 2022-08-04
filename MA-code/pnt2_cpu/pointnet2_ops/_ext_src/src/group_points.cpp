// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "group_points.h"
#include "utils.h"

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);

at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
  } else if (points.device().is_cpu()) {
    // TODO: implement group points here
    //int a=0;  //placeholder
    int b_size=points.size(0);
    int fea_num=points.size(1);
    int points_num=points.size(2);
    int center_num=idx.size(1);
    int sample_num=idx.size(2);
    int o_bias=center_num*sample_num;
    int f_index_bias, f_index,
        s_index_bias, s_index,
        o_index_bias, o_index, sample_index;
    float* input_features_ptr=points.data_ptr<float>();
    float* sampled_fea_ptr=output.data_ptr<float>();
    int* group_index_ptr=idx.data_ptr<int>();
    
    for (int b=0;b<b_size;b++){
      f_index_bias=b*fea_num*points_num;
      s_index_bias=b*center_num*sample_num;
      o_index_bias=b*fea_num*center_num*sample_num;
      for (int c=0;c<center_num;c++){
        for (int s=0;s<sample_num;s++){
          s_index=s_index_bias+c*sample_num+s;
          sample_index=group_index_ptr[s_index];

          f_index=f_index_bias+sample_index;
          o_index=o_index_bias+c*sample_num+s;
          for (int f=0;f<fea_num;f++){
            sampled_fea_ptr[o_index]=input_features_ptr[f_index];
            f_index += points_num;
            o_index += o_bias;
          }
        }
      }
    }
  }

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}
