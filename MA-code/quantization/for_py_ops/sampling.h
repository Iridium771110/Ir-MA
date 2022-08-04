#ifndef SAMPLING
#define SAMPLING
/*
sampling(points_xyz,sample_num): #对points_xyz采样sample_num个点，返回为对应index
        #(B, N, 3) tensor,int ->(B, s_n) tensor
*/
//#include <iostream>
//#include "onnxruntime_cxx_api.h"
#include "common.h"

//extern float *max_dist;

template <typename T>
struct SamplingKernel{
        private:
    Ort::CustomOpApi ort_;

        public:
    SamplingKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info): ort_(ort) {
    };
    void Compute(OrtKernelContext *context);
};

struct SamplingCustomOp: Ort::CustomOpBase<SamplingCustomOp, SamplingKernel<float>>{
    void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo *info) const{
        return new SamplingKernel<float>(api,info);
    };
    const char *GetName() const {return "onnx_sampling";};
    size_t GetInputTypeCount() const {return 2;};
    size_t GetOutputTypeCount() const {return 1;};
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index==0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        else return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    };
    ONNXTensorElementDataType GetOutputType(size_t) const {return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;};
};

#endif