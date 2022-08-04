// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sampling.h"
#include "utils.h"

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data<float>(),
                                 idx.data<int>(), output.data<float>());
  } else if (points.device().is_cpu()) {
    // TODO: implment cpu code here
    //int a=0;  //placeholder
    float *original=points.data_ptr<float>();
    float *sampled=output.data_ptr<float>();
    int *sampled_id=idx.data_ptr<int>();
    int b_size=points.size(0);
    int fea_num=points.size(1);
    int p_num=points.size(2);
    int sample_num=idx.size(1);
    int o_bias,s_bias,o_id,s_id,id,p_id;

    for (int b=0;b<b_size;b++){
      o_bias=b*fea_num*p_num;
      id=b*sample_num;
      s_bias=b*fea_num*sample_num;
      for (int s=0;s<sample_num;s++){
        p_id=sampled_id[id];
        id++;
        o_id=o_bias+p_id;
        s_id=s_bias+s;
        for (int f=0;f<fea_num;f++){
          sampled[s_id]=original[o_id];
          s_id+=sample_num;
          o_id+=p_num;
        }
      }
    }
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
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
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data<float>(),
                                      idx.data<int>(), output.data<float>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return output;
}
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data<float>(),
        tmp.data<float>(), output.data<int>());
  } else if (points.device().is_cpu()) {
    // TODO: implment cpu code here
    //int a=0;  //placeholder
    float* original=points.data_ptr<float>();
    int* sampled_id=output.data_ptr<int>();
    int b_size=points.size(0);
    int p_num=points.size(1);
    int selected_id, o_bias, s_bias, o_id, cur_id;
    float sampled_x,sampled_y,sampled_z;
    float cur_x,cur_y,cur_z,cur_max_dist,dist_x,dist_y,dist_z,dist;
    float* min_dist=new float[p_num];

    for (int b=0;b<b_size;b++){
      o_bias=b*p_num*3;
      s_bias=b*nsamples;
      for (int i=0;i<p_num;i++) min_dist[i]=1e10;
      selected_id=0;
      sampled_id[s_bias]=selected_id;
      sampled_x=original[o_bias];
      sampled_y=original[o_bias+1];
      sampled_z=original[o_bias+2];

      for (int s=1;s<nsamples;s++){
        cur_max_dist=-1.0;
        for (int p=0;p<p_num;p++){
          cur_id=o_bias+p*3;
          cur_x=original[cur_id];
          cur_y=original[cur_id+1];
          cur_z=original[cur_id+2];
          dist_x=sampled_x-cur_x;
          dist_y=sampled_y-cur_y;
          dist_z=sampled_z-cur_z;
          dist=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z;
          if (dist<min_dist[p]) min_dist[p]=dist;
          if (min_dist[p]>cur_max_dist) {cur_max_dist=min_dist[p]; selected_id=p;}
        }
        s_bias++;
        sampled_id[s_bias]=selected_id;
        o_id=o_bias+selected_id*3;
        sampled_x=original[o_id];
        sampled_y=original[o_id+1];
        sampled_z=original[o_id+2];
      }
    }
    delete []min_dist;
  }

  return output;
}
