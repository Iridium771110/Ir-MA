#ifndef BALL_QUERY_TREE
#define BALL_QUERY_TREE
/*
This file defines the operator ball query based on octree structure
ball_query(centers_xyz,points_xyz,radius,sample_num): sample sample_num points from points_xyz within the ball sapce centered in centers_xyz with radius radius，record the indices
(B, c_n, 3) tensor,(B, N, 3) tensor,float,int->(B, c_n, s_n) tensor
*/
#include "common.h"
#include "oct_tree.h"
#include <cstring>

template <typename T>
struct BallQueryKernel{
        private:
	Ort::CustomOpApi ort_;
		public:
	BallQueryKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info): ort_(ort){};
	void Compute(OrtKernelContext* context);
}; // define the operator kernel

struct BallQueryCustomOp : Ort::CustomOpBase<BallQueryCustomOp, BallQueryKernel<float>>{
	void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const{
		return new BallQueryKernel<float>(api,info);
	};
	const char* GetName() const {return "onnx_ball_query";}; // set the operator name
	size_t GetInputTypeCount() const {return 4;};
	size_t GetOutputTypeCount() const {return 1;}; // set the number of inputs and output
	ONNXTensorElementDataType GetInputType(size_t index) const{
		if (index==3) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
		else if (index==2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
		else return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	};
	ONNXTensorElementDataType GetOutputType(size_t) const{
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	}; // set the data type of inputs and output
}; // define the operator schema

void GetRangeOfSpace(const float* data, int64_t length, float &x0,float &x1,float &y0,float &y1,float &z0,float &z1){
	// set the space range of point cloud (boundary) for octree
	// Parameters: points data, number of points, minimal and maximal boundary for x,y,z
	int64_t index;
	float data_x,data_y,data_z;
	x0=1e10; x1=-1e10;
	y0=1e10; y1=-1e10;
	z0=1e10; z1=-1e10;
	for (int64_t i=0;i<length;i++){
		// traverse the points to get the boundary
		index=i*3;
		data_x=data[index];
		data_y=data[index+1];
		data_z=data[index+2];
		if (data_x<x0) x0=data_x;
		if (data_x>x1) x1=data_x;
		if (data_y<y0) y0=data_y;
		if (data_y>y0) y0=data_y;
		if (data_z<z0) z0=data_z;
		if (data_z>z1) z1=data_z;
	}
};

void MaxCommonFactor(int64_t &res, int64_t num1, int64_t num2){ 
	// get the greatest common factor with num2>num1
	// Parameters: result to store the factor, first number, second number
	if (num1>num2) {
		res=num1;
		num1=num2;
		num2=res;
	}
	while((num2%num1)!=0){
		res=num2%num1;
		num2=num1;
		num1=res;
	}
	res=num1;
};

template <typename T>
void BallQueryKernel<T>::Compute(OrtKernelContext *context){
	// implement the operator

	// get the information of inputs
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

	// set the information of output
	std::vector<int64_t> dim_o(3);
	dim_o[0]=b_size; dim_o[1]=center_num; dim_o[2]=sample_num;
	OrtValue* output=ort_.KernelContext_GetOutput(context,0,dim_o.data(),dim_o.size());
	int* sampled_index_ptr=ort_.GetTensorMutableData<int>(output);
	OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
	ort_.ReleaseTensorTypeAndShapeInfo(output_info);

	// operation part
	float reso,x0,x1,y0,y1,z0,z1;
	int m_label,label;
	int* index_data=nullptr;
	int labels[27];
	int* p_label=new int[points_num];
	int* repeated=new int[points_num]; 
	int64_t c_index_bias,c_index,
			p_index_bias,p_index,
			s_index_bias,s_index, s_index_m, ll,m_index,stride;
	double  center_x,point_x,dist_x,
			center_y,point_y,dist_y,
			center_z,point_z,dist_z, dist2;
	OCT_TREE::Point point;
	reso=2*radius; // this resolution ensures that the length of leaf node larger than radius, so the ball space can be included in 26 neighborhoods
	
	for (int64_t b=0;b<b_size;b++){ // loop for batch size
		std::memset(repeated,0,sizeof(int)*points_num);
		p_index_bias=b*points_num*3;
		c_index_bias=b*center_num*3;
		s_index_bias=b*center_num*sample_num;
		GetRangeOfSpace(points_ptr+p_index_bias,points_num,x0,x1,y0,y1,z0,z1);
		OCT_TREE::Oct_Tree oct_tree(reso,x0,x1,y0,y1,z0,z1); // set the space range and initialize a octree

		m_label=0;
		for (int64_t i=0;i<points_num;i++){
			// create the tree traversing the points
			p_index=p_index_bias+i*3;
			point.x=points_ptr[p_index];
			point.y=points_ptr[p_index+1];
			point.z=points_ptr[p_index+2];
			label=oct_tree.Creat_Tree(point);
			p_label[i]=label;
			if (m_label<label) m_label=label;
		}
		m_label++; // record the number of valid leaf node

		int node_p_num[m_label];
		std::memset(node_p_num,0,sizeof(int)*m_label);
		for (int64_t i=0;i<points_num;i++) node_p_num[p_label[i]]++; // record the number of points for each valid leaf node
		std::vector<std::vector<int>> node_p_index(m_label);
		for (int64_t i=0;i<m_label;i++) node_p_index[i].resize(node_p_num[i]);
		for (int64_t i=0;i<points_num;i++){
			label=p_label[i];
			node_p_num[label]--;
			node_p_index[label][node_p_num[label]]=i;
		} // record the indices of contained points for each valid leaf node

		for (int64_t c=0;c<center_num;c++){ // loop for centers
			c_index=c_index_bias+c*3;
			s_index=s_index_bias+c*sample_num;
			s_index_m=s_index+sample_num;
			center_x=center_ptr[c_index];
			center_y=center_ptr[c_index+1];
			center_z=center_ptr[c_index+2];

			// record the labels of blocks (center self and the 26 neighbors)
			point.x=center_x; point.y=center_y; point.z=center_z;
			labels[0]=oct_tree.Get_position_label(point);

			point.x=center_x+radius; point.y=center_y; point.z=center_z;
			labels[1]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y+radius; point.z=center_z;
			labels[2]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y; point.z=center_z+radius;
			labels[3]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y; point.z=center_z;
			labels[4]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y-radius; point.z=center_z;
			labels[5]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y; point.z=center_z-radius;
			labels[6]=oct_tree.Get_position_label(point);

			point.x=center_x+radius; point.y=center_y+radius; point.z=center_z;
			labels[7]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y+radius; point.z=center_z+radius;
			labels[8]=oct_tree.Get_position_label(point);
			point.x=center_x+radius; point.y=center_y; point.z=center_z+radius;
			labels[9]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y-radius; point.z=center_z;
			labels[10]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y-radius; point.z=center_z-radius;
			labels[11]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y; point.z=center_z-radius;
			labels[12]=oct_tree.Get_position_label(point);

			point.x=center_x+radius; point.y=center_y-radius; point.z=center_z;
			labels[13]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y+radius; point.z=center_z-radius;
			labels[14]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y; point.z=center_z+radius;
			labels[15]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y+radius; point.z=center_z;
			labels[16]=oct_tree.Get_position_label(point);
			point.x=center_x; point.y=center_y-radius; point.z=center_z+radius;
			labels[17]=oct_tree.Get_position_label(point);
			point.x=center_x+radius; point.y=center_y; point.z=center_z-radius;
			labels[18]=oct_tree.Get_position_label(point);
			
			point.x=center_x+radius; point.y=center_y+radius; point.z=center_z-radius;
			labels[19]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y+radius; point.z=center_z+radius;
			labels[20]=oct_tree.Get_position_label(point);
			point.x=center_x+radius; point.y=center_y-radius; point.z=center_z+radius;
			labels[21]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y-radius; point.z=center_z+radius;
			labels[22]=oct_tree.Get_position_label(point);
			point.x=center_x+radius; point.y=center_y-radius; point.z=center_z-radius;
			labels[23]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y+radius; point.z=center_z-radius;
			labels[24]=oct_tree.Get_position_label(point);

			point.x=center_x+radius; point.y=center_y+radius; point.z=center_z+radius;
			labels[25]=oct_tree.Get_position_label(point);
			point.x=center_x-radius; point.y=center_y-radius; point.z=center_z-radius;
			labels[26]=oct_tree.Get_position_label(point);

//==========================================================================================
// this sampling version demands more time cost but gains better precision
// It first extracts all the feasible points and then samples with a stride (means more evenly)
			// ll=0;
			// for (int j=0;j<27;j++){
			// 	label=labels[j];
			// 	if (label<0) continue;
			// 	if (repeated[label]!=c+1){
			// 		repeated[label]=c+1;
			// 		//std::cout<<label<<std::endl;
			// 		std::memcpy(p_label+ll,node_p_index[label].data(),sizeof(int)*node_p_index[label].size());
			// 		ll+=node_p_index[label].size();
			// 	} // record indices of all the points in the 27 blocks above without repetition
			// }
			// // if (ll>sample_num) {
			// // 	stride=ll/sample_num+1;
			// // 	MaxCommonFactor(m_index,stride,ll);
			// // 	m_index=stride*ll/m_index; 
			// // }	// set stride and calculate the least common multiple for number of searching points and stride
			// // else {
			// // 	stride=1;
			// // 	m_index=sample_num;
			// // } // set stride and calculate the least common multiple for number of searching points and stride
			// for (int64_t j=0;j<ll;j++){ // loop for searching
			// 	if (s_index==s_index_m) break; // check if the required number is reached 
			// 	//label=p_label[(j*stride)%ll+(j*stride)/m_index]; // traverse cyclically with stride
			// 	label=p_label[j]; // traverse sequentially
			// 	p_index=p_index_bias+label*3;
			// 	point_x=points_ptr[p_index];
			// 	point_y=points_ptr[p_index+1];
			// 	point_z=points_ptr[p_index+2];
			// 	dist_x=center_x-point_x;
			// 	dist_y=center_y-point_y;
			// 	dist_z=center_z-point_z;
			// 	dist2=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z;
			// 	if (r2>dist2){
			// 		sampled_index_ptr[s_index]=label; // record the index of applied point
			// 		s_index++;
			// 	}
			// }
			
//=================================================================================
// this sampling version demands less time cost but gains worse precision
// It samples directly while searching the feasible points (means a pre-defined trend for sampled points)
			for (int j=0;j<27;j++){ // loop for clocks
				if (s_index==s_index_m) break; // check if the required number is reached 
				label=labels[j];
				if ((label>=0)&&(repeated[label]!=c+1)){ // searching without repetition
					repeated[label]=c+1;
					m_index=node_p_index[label].size();
					index_data=node_p_index[label].data();
					for (int64_t i=0;i<m_index;i++){ // loop for searching of one block
						if (s_index==s_index_m) break; // check if the required number is reached 
						p_index=p_index_bias+index_data[i]*3;
						point_x=points_ptr[p_index];
						point_y=points_ptr[p_index+1];
						point_z=points_ptr[p_index+2];
						dist_x=center_x-point_x;
						dist_y=center_y-point_y;
						dist_z=center_z-point_z;
						dist2=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z;
						if (r2>dist2){
							sampled_index_ptr[s_index]=index_data[i]; // record the index of applied point
							s_index++;
						}
					}
					index_data=nullptr;
				}
			}
//===================================================================================

			ll=sampled_index_ptr[s_index_bias+c*sample_num];
	 		for (int64_t s=s_index;s<s_index_m;s++) sampled_index_ptr[s]=ll; // repeat the first point until the required number is reached if sampled points are not enough
			//if(ll>points_num) std::cout<<ll<<' '<<m_index<<' '<<stride<<std::endl;
		}
		
	}

	delete []p_label;
	delete []repeated;
};

#endif