#ifndef SAMPLING
#define SAMPLING
/*
This file defines the operator sampling
sampling(points_xyz,sample_num): sample sample_num points in points_xyz and record the indices of sampled points
(B, N, 3) tensor,int ->(B, s_n) tensor
*/

#include "common.h"

template <typename T>
struct SamplingKernel{
        private:
    Ort::CustomOpApi ort_;

        public:
    SamplingKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info): ort_(ort) {
    };
    void Compute(OrtKernelContext *context);
}; // define the operator kernel

struct SamplingCustomOp: Ort::CustomOpBase<SamplingCustomOp, SamplingKernel<float>>{
    void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo *info) const{
        return new SamplingKernel<float>(api,info);
    };
    const char *GetName() const {return "onnx_sampling";}; // set the operator name
    size_t GetInputTypeCount() const {return 2;};
    size_t GetOutputTypeCount() const {return 1;}; // set the number of inputs and output
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index==0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        else return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    };
    ONNXTensorElementDataType GetOutputType(size_t) const {return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;}; // set the data type of inputs and output
}; // define the operator schema

template<typename T>
void SamplingKernel<T>::Compute(OrtKernelContext *context){
    // implement the operator using furthest point sampling algorithm

    // get the information of inputs
    const OrtValue* input_data=ort_.KernelContext_GetInput(context,0);
    const T* original=reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_data));
    OrtTensorDimensions dims_i(ort_,input_data);
    const int64_t b_size=dims_i[0];
    const int64_t o_p_num=dims_i[1];

    const OrtValue* input_2=ort_.KernelContext_GetInput(context,1);
    const int64_t* to_sample_num_ptr=reinterpret_cast<const int64_t*>(ort_.GetTensorData<int64_t>(input_2));
    const int64_t to_sample_num=to_sample_num_ptr[0];
    const int64_t s_p_num=to_sample_num;

    // set the information of output
    std::vector<int64_t> dims_output(2);
    dims_output[0]=b_size; dims_output[1]=s_p_num;
    OrtValue *output=ort_.KernelContext_GetOutput(context,0,dims_output.data(),dims_output.size());
    int *sampled=ort_.GetTensorMutableData<int>(output);
    OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // operation part
    int64_t s_index_bias,o_index_bias,index,selected_id;
    float *max_dist=new float[o_p_num];
    float sampled_x,sampled_y,sampled_z,cur_x,cur_y,cur_z,dist_x,dist_y,dist_z;
    float dist,cur_max_dist;

    for (int64_t batch_num=0;batch_num<b_size;batch_num++){ // loop for batch size
        s_index_bias=batch_num*s_p_num;
        o_index_bias=batch_num*o_p_num*3;
        for (int64_t i=0;i<o_p_num;i++) max_dist[i]=1e10; // initialize
        sampled_x=original[o_index_bias];
        sampled_y=original[o_index_bias+1]; 
        sampled_z=original[o_index_bias+2];
        sampled[s_index_bias]=0; // take the first point as the first sampled point

        for (int64_t i=1;i<s_p_num;i++){ // loop for sample number
            cur_max_dist=-1.0;
            for (int64_t j=0;j<o_p_num;j++){ // loop for original points set
                index=o_index_bias+j*3;
                cur_x=original[index]; cur_y=original[index+1]; cur_z=original[index+2];

                dist_x=cur_x-sampled_x; dist_y=cur_y-sampled_y; dist_z=cur_z-sampled_z;
                dist=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z; // calculate the distance between the last sampled point and the current point
                if (dist<max_dist[j]) max_dist[j]=dist; // update for each point the minimal distance to the sampled points
                if (max_dist[j]>cur_max_dist) {cur_max_dist=max_dist[j]; selected_id=j;} // update the maximal distance and the index of point for current search
            }
            
            index=s_index_bias+i;
            sampled[index]=selected_id; 
            index=o_index_bias+selected_id*3;
            sampled_x=original[index];
            sampled_y=original[index+1];
            sampled_z=original[index+2]; // take the point with maximal distance in current search as the sampled point
        }
    }
    delete []max_dist;
};
#endif