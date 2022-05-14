#ifndef BALL_QUERY_TREE
#define BALL_QUERY_TREE

#include "common.h"
#include "ball_query_third_kd.h"
#include <cstring>

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

template <typename T>
void BallQueryKernel<T>::Compute(OrtKernelContext *context){
	const OrtValue* center_tensor=ort_.KernelContext_GetInput(context,0);
	const T* center_ptr=reinterpret_cast<const T*>(ort_.GetTensorData<T>(center_tensor));
	OrtTensorDimensions dim_c(ort_,center_tensor);
	const int64_t b_size=dim_c[0];
	const int64_t center_num=dim_c[1];

	const OrtValue* points_tensor=ort_.KernelContext_GetInput(context,1);
	const T* points_ptr=reinterpret_cast<const T*>(ort_.GetTensorData<T>(points_tensor));
	OrtTensorDimensions dim_p(ort_,points_tensor);
	const int64_t points_num=dim_p[1];

	const OrtValue* radius_tensor=ort_.KernelContext_GetInput(context,2);
	const double* radius_ptr=reinterpret_cast<const double*>(ort_.GetTensorData<double>(radius_tensor));
	const double radius=radius_ptr[0];
	const double r2=radius*radius;

	const OrtValue* sample_num_tensor=ort_.KernelContext_GetInput(context,3);
	const int64_t* sample_num_ptr=reinterpret_cast<const int64_t*>(ort_.GetTensorData<int64_t>(sample_num_tensor));
	const int64_t sample_num=sample_num_ptr[0];

	std::vector<int64_t> dim_o(3);
	dim_o[0]=b_size; dim_o[1]=center_num; dim_o[2]=sample_num;
	OrtValue* output=ort_.KernelContext_GetOutput(context,0,dim_o.data(),dim_o.size());
	int* sampled_index_ptr=ort_.GetTensorMutableData<int>(output);
	OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
	ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    cpp_knn_radius(points_ptr,points_num,3,center_ptr,center_num,sample_num,radius,sampled_index_ptr);

};
#endif