#ifndef GROUPING
#define GROUPING
/*
This file defines the operator grouping
grouping(full_features_map,grouping_index): extract the point features from full_features_map according to the point index in grouping_index
(B, C, N) tensor,(B, c_n, s_n) tensor ->(B, C, c_n, s_n) tensor
*/
#include "common.h"

template<typename T>
struct GroupingKernel{
		private:
	Ort::CustomOpApi ort_;
		public:   
	GroupingKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info): ort_(ort){};
	void Compute(OrtKernelContext *context);
}; // define the operator kernel

struct GroupingCustomOp : Ort::CustomOpBase<GroupingCustomOp, GroupingKernel<float>>{
	void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const{
		return new GroupingKernel<float>(api,info);
	};
	const char* GetName() const {return "onnx_grouping";}; // set the operator name
	size_t GetInputTypeCount() const {return 2;};
	size_t GetOutputTypeCount() const {return 1;}; // set the number of inputs and output
	ONNXTensorElementDataType GetInputType(size_t index) const {
		if (index==0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		else return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	};
	ONNXTensorElementDataType GetOutputType(size_t) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	}; // set the data type of inputs and output
}; // define the operator schema

template<typename T>
void GroupingKernel<T>::Compute(OrtKernelContext *context){
	// implement the operator

	// get the information of inputs
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

	// set the information of output
	std::vector<int64_t> output_dim(4);
	output_dim[0]=b_size; output_dim[1]=fea_num; output_dim[2]=center_num; output_dim[3]=sample_num;
	OrtValue* output=ort_.KernelContext_GetOutput(context,0,output_dim.data(),output_dim.size());
	T* sampled_fea_ptr=ort_.GetTensorMutableData<T>(output);
	OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
	ort_.ReleaseTensorTypeAndShapeInfo(output_info);

	// operation part
	const int64_t o_bias=center_num*sample_num;
	int64_t f_index_bias,f_index,f_index_base,
			s_index_bias,s_index,
			o_index_bias,o_index,sample_index;

	//============================================================
	// original access sequence
	// for (int64_t b=0;b<b_size;b++){ // loop for batch size
	// 	f_index_bias=b*fea_num*points_num;
	// 	s_index_bias=b*center_num*sample_num;
	// 	o_index_bias=b*fea_num*center_num*sample_num;
	// 	for (int64_t c=0;c<center_num;c++){ // loop for ball query centers
	// 		for (int64_t s=0;s<sample_num;s++){ // loop for indices
	// 			s_index=s_index_bias+c*sample_num+s;
	// 			sample_index=group_index_ptr[s_index]; // get the index of sampled point

	// 			f_index=f_index_bias+sample_index;
	// 			o_index=o_index_bias+c*sample_num+s;
	// 			for (int64_t f=0;f<fea_num;f++){ // loop for features
	// 				sampled_fea_ptr[o_index]=input_features_ptr[f_index]; // featrue transmission
	// 				f_index += points_num;
	// 				o_index += o_bias;
	// 			}
	// 		}
	// 	}
	// }

	//============================================================
	// other access sequence with higher cache hit ratio
	for (int64_t b=0;b<b_size;b++){ // loop for batch size
		f_index_bias=b*fea_num*points_num;
		s_index_bias=b*center_num*sample_num;
		o_index_bias=b*fea_num*center_num*sample_num;
		for (int64_t f=0;f<fea_num;f++){ // loop for features
			f_index_base=f_index_bias+f*points_num;
			o_index=o_index_bias+f*o_bias;
			for (int64_t c=0;c<center_num;c++){ // loop for ball query centers
				s_index=s_index_bias+c*sample_num;
				for (int64_t s=0;s<sample_num;s++){ // loop for indices
					sample_index=group_index_ptr[s_index]; // get the index of sampled point
					s_index++;
					f_index=f_index_base+sample_index;
					sampled_fea_ptr[o_index]=input_features_ptr[f_index]; // feature transmission
					o_index++;
				}
			}		
		}
	}

	//=============================================================
	// multi-threading based on the access sequence with higher cache hit ratio
	// for (int64_t b=0;b<b_size;b++){ // loop for batch size
	// 	int64_t p_f_index_bias=b*fea_num*points_num;
	// 	int64_t p_s_index_bias=b*center_num*sample_num;
	// 	int64_t p_o_index_bias=b*fea_num*center_num*sample_num;

	// 	#pragma omp parallel for num_threads(6) // set number of threads for multi-threading
	// 	for (int64_t f=0;f<fea_num;f++){ // loop for features
	// 		const T* p_f_ptr=input_features_ptr+p_f_index_bias+f*points_num;
	// 		T* p_o_ptr=sampled_fea_ptr+p_o_index_bias+f*o_bias; // new ptr for multi-threading adaption
			
	// 		for (int64_t c=0;c<center_num;c++){ // loop for ball query centers
	// 			const int* p_s_ptr=group_index_ptr+p_s_index_bias+c*sample_num;
	// 			for (int64_t s=0;s<sample_num;s++){ // loop for indices
	// 				int p_sample_index=p_s_ptr[s]; // get the index of sampled point
	// 				*p_o_ptr=p_f_ptr[p_sample_index]; // feature transmission
	// 				p_o_ptr+=1;
	// 			}
	// 		}
	// 	}
	// }
};

#endif