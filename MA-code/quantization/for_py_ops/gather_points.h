#ifndef GATHER_POINTS
#define GATHER_POINTS
/*
gather_points(points_xyz,gather_index): #对points_xyz取对应gather_index的点，返回为所取点
        #(B, C, N) tensor,(B, s_n) tensor ->(B, C, s_n) tensor
*/
#include "common.h"

template<typename T>
struct GatherPointsKernel{
        private:
    Ort::CustomOpApi ort_;
        public:
	GatherPointsKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info): ort_(ort) {};
	void Compute(OrtKernelContext *context);
};

struct GatherPointsCustomOp : Ort::CustomOpBase<GatherPointsCustomOp, GatherPointsKernel<float>>{
	void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo *info) const{
		return new GatherPointsKernel<float>(api,info);
	};
	const char* GetName() const {return "onnx_gather_points";};
	size_t GetInputTypeCount() const {return 2;};
	size_t GetOutputTypeCount() const {return 1;};
	ONNXTensorElementDataType GetInputType(size_t index) const {
		if (index==0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		else return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	};
	ONNXTensorElementDataType GetOutputType(size_t) const {return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;};
};


#endif