// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());
  } else if (new_xyz.device().is_cpu()) {
    // TODO: implement ball query here
    //int a=0;  //placeholder
    float* points_ptr=xyz.data_ptr<float>();
    float* center_ptr=new_xyz.data_ptr<float>();
    int* sampled_index_ptr=idx.data_ptr<int>();
    float r2=radius*radius;
    int b_size=xyz.size(0);
    int points_num=xyz.size(1);
    int center_num=new_xyz.size(1);
    int sample_num=nsample;
    int c_index_bias, c_index,
        p_index_bias, p_index,
        s_index_bias, s_index, s_index_m, ll;
    float center_x, center_y, center_z,
          point_x, point_y, point_z,
          dist_x, dist_y, dist_z, dist2;

    for (int b=0;b<b_size;b++){
      c_index_bias=b*center_num*3;
      p_index_bias=b*points_num*3;
      s_index_bias=b*center_num*sample_num;
      for (int c=0;c<center_num;c++){
        c_index=c_index_bias+c*3;
        s_index=s_index_bias+c*sample_num;
        s_index_m=s_index+sample_num;
        center_x=center_ptr[c_index];
        center_y=center_ptr[c_index+1];
        center_z=center_ptr[c_index+2];
        for (int p=0;p<points_num;p++){
          if (s_index==s_index_m) break;
          p_index=p_index_bias+p*3;
          point_x=points_ptr[p_index];
          point_y=points_ptr[p_index+1];
          point_z=points_ptr[p_index+2];
          dist_x=point_x-center_x;
          dist_y=point_y-center_y;
          dist_z=point_z-center_z;
          dist2=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z;
          if (r2>dist2){
            sampled_index_ptr[s_index]=int(p);
            s_index++;
          }
        }
        // std::memset(sampled_index_ptr+s_index,0,sizeof(int)*(s_index_m-s_index));
        ll=sampled_index_ptr[s_index_bias+c*sample_num];
        for (int64_t s=s_index;s<s_index_m;s++) sampled_index_ptr[s]=ll;
      }
    }
  }

  return idx;
}
