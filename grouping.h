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

template<typename T>
void GroupingKernel<T>::Compute(OrtKernelContext *context){
	const OrtValue* input_features_tensor=ort_.KernelContext_GetInput(context,0);
	const T* input_features_ptr=reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_features_tensor));
	OrtTensorDimensions dims_f(ort_,input_features_tensor);
	const int64_t b_size=dims_f[0];
	const int64_t fea_num=dims_f[1];
	const int64_t points_num=dims_f[2];

	const OrtValue* group_index_tensor=ort_.KernelContext_GetInput(context,1);
	const int* group_index_ptr=reinterpret_cast<const int*>(ort_.GetTensorData<int>(group_index_tensor));
	OrtTensorDimensions dims_i(ort_,group_index_tensor);
	const int64_t center_num=dims_i[1];
	const int64_t sample_num=dims_i[2];

	std::vector<int64_t> output_dim(4);
	output_dim[0]=b_size; output_dim[1]=fea_num; output_dim[2]=center_num; output_dim[3]=sample_num;
	OrtValue* output=ort_.KernelContext_GetOutput(context,0,output_dim.data(),output_dim.size());
	T* sampled_fea_ptr=ort_.GetTensorMutableData<T>(output);
	OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
	ort_.ReleaseTensorTypeAndShapeInfo(output_info);

	const int64_t o_bias=center_num*sample_num;
	int64_t f_index_bias,f_index,f_index_base,
			s_index_bias,s_index,
			o_index_bias,o_index,sample_index;

	// for (int64_t b=0;b<b_size;b++){
	// 	f_index_bias=b*fea_num*points_num;
	// 	s_index_bias=b*center_num*sample_num;
	// 	o_index_bias=b*fea_num*center_num*sample_num;
	// 	for (int64_t c=0;c<center_num;c++){
	// 		for (int64_t s=0;s<sample_num;s++){
	// 			s_index=s_index_bias+c*sample_num+s;
	// 			sample_index=group_index_ptr[s_index];

	// 			f_index=f_index_bias+sample_index;
	// 			o_index=o_index_bias+c*sample_num+s;
	// 			for (int64_t f=0;f<fea_num;f++){
	// 				sampled_fea_ptr[o_index]=input_features_ptr[f_index];
	// 				f_index += points_num;
	// 				o_index += o_bias;
	// 			}
	// 		}
	// 	}
	// }


	//------- try other order
	for (int64_t b=0;b<b_size;b++){
		f_index_bias=b*fea_num*points_num;
		s_index_bias=b*center_num*sample_num;
		o_index_bias=b*fea_num*center_num*sample_num;
		for (int64_t f=0;f<fea_num;f++){
			f_index_base=f_index_bias+f*points_num;
			o_index=o_index_bias+f*o_bias;
			for (int64_t c=0;c<center_num;c++){
				s_index=s_index_bias+c*sample_num;
				for (int64_t s=0;s<sample_num;s++){
					sample_index=group_index_ptr[s_index];
					s_index++;
					f_index=f_index_base+sample_index;
					sampled_fea_ptr[o_index]=input_features_ptr[f_index];
					o_index++;
				}
			}		
		}
	}
};

#endif