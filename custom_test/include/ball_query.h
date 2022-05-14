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

	int64_t c_index_bias,c_index,
			p_index_bias,p_index,
			s_index_bias,s_index, s_index_m, ll;
	double  center_x,point_x,dist_x,
			center_y,point_y,dist_y,
			center_z,point_z,dist_z, dist2;

	for (int64_t b=0;b<b_size;b++){
		c_index_bias=b*center_num*3;
		p_index_bias=b*points_num*3;
		s_index_bias=b*center_num*sample_num;
		for (int64_t c=0;c<center_num;c++){
			c_index=c_index_bias+c*3;
			s_index=s_index_bias+c*sample_num;
			s_index_m=s_index+sample_num;
			center_x=center_ptr[c_index];
			center_y=center_ptr[c_index+1];
			center_z=center_ptr[c_index+2];
			for (int64_t p=0;p<points_num;p++){
				if (s_index==s_index_m) break;
				p_index=p_index_bias+p*3;
				point_x=points_ptr[p_index];
				point_y=points_ptr[p_index+1];
				point_z=points_ptr[p_index+2];
				dist_x=point_x-center_x;
				dist_y=point_y-center_y;
				dist_z=point_z-center_z;
				dist2=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z;
				if (r2>dist2){
					sampled_index_ptr[s_index]=int(p);
					s_index++;
				}
			}
			// std::memset(sampled_index_ptr+s_index,0,sizeof(int)*(s_index_m-s_index));
			ll=sampled_index_ptr[s_index_bias+c*sample_num];
			for (int64_t s=s_index;s<s_index_m;s++) sampled_index_ptr[s]=ll;
		}
	}
	
};
#endif
