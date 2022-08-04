#ifndef GROUPING
#define GROUPING
/*
grouping(full_features_map,grouping_index): #对full_features_map按照grouping_index取对应点，返回为所取点（特征）
        #(B, C, N) tensor,(B, c_n, s_n) tensor ->(B, C, c_n, s_n) tensor
*/
#include "common.h"


template<typename T>
struct GroupingKernel{
		private:
	Ort::CustomOpApi ort_;
		public:   
	GroupingKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info): ort_(ort){};
	void Compute(OrtKernelContext *context);
};

struct GroupingCustomOp : Ort::CustomOpBase<GroupingCustomOp, GroupingKernel<float>>{
	void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const{
		return new GroupingKernel<float>(api,info);
	};
	const char* GetName() const {return "onnx_grouping";};
	size_t GetInputTypeCount() const {return 2;};
	size_t GetOutputTypeCount() const {return 1;};
	ONNXTensorElementDataType GetInputType(size_t index) const {
		if (index==0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		else return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	};
	ONNXTensorElementDataType GetOutputType(size_t) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	};
};


#endif