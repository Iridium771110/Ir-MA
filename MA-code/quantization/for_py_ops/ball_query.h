#ifndef BALL_QUERY
#define BALL_QUERY
/*
ball_query(centers_xyz,points_xyz,radius,sample_num): #对points_xyz以centers_xyz为中心以radius为半径取最多sample_num个点，返回为对应index
        #(B, c_n, 3) tensor,(B, N, 3) tensor,float,int->(B, c_n, s_n) tensor
*/
#include "common.h"

template <typename T>
struct BallQueryKernel{
        private:
	Ort::CustomOpApi ort_;
		public:
	BallQueryKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info): ort_(ort){};
	void Compute(OrtKernelContext* context);
};

struct BallQueryCustomOp : Ort::CustomOpBase<BallQueryCustomOp, BallQueryKernel<float>>{
	void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const{
		return new BallQueryKernel<float>(api,info);
	};
	const char* GetName() const {return "onnx_ball_query";};
	size_t GetInputTypeCount() const {return 4;};
	size_t GetOutputTypeCount() const {return 1;};
	ONNXTensorElementDataType GetInputType(size_t index) const{
		if (index==3) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
		else if (index==2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
		else return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	};
	ONNXTensorElementDataType GetOutputType(size_t) const{
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	};
};


#endif
