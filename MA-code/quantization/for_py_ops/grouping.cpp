#include "grouping.h"

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

	//-----try paralell
	
	// for (int64_t b=0;b<b_size;b++){
	// 	int64_t p_f_index_bias=b*fea_num*points_num;
	// 	int64_t p_s_index_bias=b*center_num*sample_num;
	// 	int64_t p_o_index_bias=b*fea_num*center_num*sample_num;

	// 	//std::cout<<fea_num<<std::endl;
	// 	#pragma omp parallel for num_threads(4)
	// 	for (int64_t f=0;f<fea_num;f++){
	// 		const T* p_f_ptr=input_features_ptr+p_f_index_bias+f*points_num;
	// 		T* p_o_ptr=sampled_fea_ptr+p_o_index_bias+f*o_bias;
			
	// 		//copy_features(center_num,sample_num,p_s_index_bias,group_index_ptr,p_f_ptr,p_o_ptr);
	// 		for (int64_t c=0;c<center_num;c++){
	// 			const int* p_s_ptr=group_index_ptr+p_s_index_bias+c*sample_num;
	// 			for (int64_t s=0;s<sample_num;s++){
	// 				int p_sample_index=p_s_ptr[s];
	// 				*p_o_ptr=p_f_ptr[p_sample_index];
	// 				p_o_ptr+=1;
	// 			}
	// 		}
	// 	}
	// }
};

template void GroupingKernel<float>::Compute(OrtKernelContext *context);