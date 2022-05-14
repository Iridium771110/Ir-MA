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

template<typename T>
void GatherPointsKernel<T>::Compute(OrtKernelContext *context){
	const OrtValue* input_features_tensor=ort_.KernelContext_GetInput(context,0);
	const T* input_features_ptr=reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_features_tensor));
	OrtTensorDimensions dims_f(ort_,input_features_tensor);
	const int64_t b_size=dims_f[0];
	const int64_t fea_num=dims_f[1];
	const int64_t points_num=dims_f[2];

	const OrtValue* input_sample_index_tensor=ort_.KernelContext_GetInput(context,1);
	const int* input_sample_index_ptr=reinterpret_cast<const int*>(ort_.GetTensorData<int>(input_sample_index_tensor));
	OrtTensorDimensions dims_s(ort_,input_sample_index_tensor);
	const int64_t sample_num=dims_s[1];

	if (points_num<sample_num) throw "too few points to sample, number of points less than the number should be sampled!";

	std::vector<int64_t> output_dim(3);
	output_dim[0]=b_size; output_dim[1]=fea_num; output_dim[2]=sample_num;
	OrtValue* output=ort_.KernelContext_GetOutput(context,0,output_dim.data(),output_dim.size());
	T* sampled_fea_ptr=ort_.GetTensorMutableData<T>(output);
	OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
	ort_.ReleaseTensorTypeAndShapeInfo(output_info);

	int64_t f_index_bias,f_index,f_index_base,
			o_index_bias,o_index,
			s_index_bias,s_index,
			sample_index;

	for (int64_t b=0;b<b_size;b++){
		f_index_bias=b*fea_num*points_num;
		o_index_bias=b*fea_num*sample_num;
		s_index_bias=b*sample_num;
		for (int64_t i=0;i<sample_num;i++){
			s_index=s_index_bias+i;
			sample_index=input_sample_index_ptr[s_index];

			f_index=f_index_bias+sample_index;
			o_index=o_index_bias+i;
			for (int64_t f=0;f<fea_num;f++){
				sampled_fea_ptr[o_index]=input_features_ptr[f_index];
				o_index += sample_num;
				f_index += points_num;
			}
		}
	}

	//try other order
	// for (int64_t b=0;b<b_size;b++){
	// 	f_index_bias=b*fea_num*points_num;
	// 	o_index_bias=b*fea_num*sample_num;
	// 	s_index_bias=b*sample_num;
	// 	for (int64_t f=0;f<fea_num;f++){
	// 		f_index_base=f_index_bias+f*points_num;
	// 		o_index=o_index_bias+f*sample_num;
	// 		s_index=s_index_bias;
	// 		for (int64_t i=0;i<sample_num;i++){
	// 			sample_index=input_sample_index_ptr[s_index];
	// 			s_index++;
	// 			f_index=f_index_bias+sample_index;
	// 			sampled_fea_ptr[o_index]=input_features_ptr[f_index];
	// 			o_index++;
	// 		}
	// 	}
	// }

};

#endif