#ifndef TEST_CUSTOM_FPS_ONNX
#define TEST_CUSTOM_FPS_ONNX

/*
This code define the kernel and the schema of fps custom operator
And implement the fps algorithm under compute function, which means the implementation under ONNX Runtime
*/

#include <iostream>
#include "onnxruntime_cxx_api.h"

struct Input{
    const char *name;
    std::vector<int64_t> dims;
    std::vector<float> values;
};
struct Input1{
    const char *name;
    std::vector<int64_t> dims;
    std::vector<float> values;
};
struct Input2{
    const char *name;
    std::vector<int64_t> dims;
    std::vector<int64_t> values;
};
struct Input3{
    const char *name;
    int64_t value;
};

struct OrtTensorDimensions : std::vector<int64_t> {
    // define dimensional information of tensor
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value){
        OrtTensorTypeAndShapeInfo *info=ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

template <typename T>
struct FpsKernel{
    // define the Kernel of the custom operator
        private:
    Ort::CustomOpApi ort_;
    int64_t tt;

        public:
    FpsKernel(Ort::CustomOpApi ort, const OrtKernelInfo *info): ort_(ort) {
        tt = ort_.KernelInfoGetAttribute<int64_t>(info, "epsilon"); //get the external parameter
    };
    void Compute(OrtKernelContext *context); // to implement the algrithm
};

struct FpsCustomOp: Ort::CustomOpBase<FpsCustomOp, FpsKernel<float>>{
    // define the schema of the custom operator
    void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo *info) const{
        return new FpsKernel<float>(api,info);
    };
    const char *GetName() const {return "test_fps_node";}; // set the name of the operator
    size_t GetInputTypeCount() const {return 3;}; // set the number of inputs
    size_t GetOutputTypeCount() const {return 1;}; // set the number of outputs
    ONNXTensorElementDataType GetInputType(size_t index) const {
        // set the data type of inputs
        if (index==0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        else if (index==1) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        else return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        };
    ONNXTensorElementDataType GetOutputType(size_t) const {return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}; // set the data type of outputs
};

template<typename T>
void FpsKernel<T>::Compute(OrtKernelContext *context){
    // implement the fps algorithm under ONNX Runtime

    const OrtValue* input_data=ort_.KernelContext_GetInput(context,0);
    const OrtValue* output_table=ort_.KernelContext_GetInput(context,1);
    const T* original=reinterpret_cast<const T*>(ort_.GetTensorData<T>(input_data));
    OrtTensorDimensions dims_i(ort_,input_data);
    const int64_t b_size=dims_i[0];
    const int64_t o_p_num=dims_i[2]; // initialize the inputs data

    OrtTensorDimensions dims_o(ort_,output_table);
    OrtValue *output=ort_.KernelContext_GetOutput(context,0,dims_o.data(),dims_o.size());
    T *sampled=ort_.GetTensorMutableData<T>(output);
    const int64_t s_p_num=dims_o[2];
    OrtTensorTypeAndShapeInfo* output_info=ort_.GetTensorTypeAndShape(output);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info); // initialize the output data
    
    // implement the fps algorithm
    int64_t s_index_bias,o_index_bias,index,selected_id;
    float *max_dist=new float[o_p_num];
    float sampled_x,sampled_y,sampled_z,cur_x,cur_y,cur_z,dist_x,dist_y,dist_z;
    float dist,cur_max_dist;
    for (int64_t batch_num=0;batch_num<b_size;batch_num++){ // loop for batch size
        s_index_bias=batch_num*s_p_num*3;
        o_index_bias=batch_num*o_p_num*3;
        for (int64_t i=0;i<o_p_num;i++) max_dist[i]=1e10;
        sampled_x=original[o_index_bias];
        sampled_y=original[o_index_bias+1]; 
        sampled_z=original[o_index_bias+2];

        // sampled[s_index_bias]=sampled_x;
        // sampled[s_index_bias+1]=sampled_y;
        // sampled[s_index_bias+2]=sampled_z;

        // sampled_x=original[o_index_bias];
        // sampled_y=original[o_index_bias+o_p_num]; 
        // sampled_z=original[o_index_bias+2*o_p_num];

        sampled[s_index_bias]=sampled_x;
        sampled[s_index_bias+s_p_num]=sampled_y;
        sampled[s_index_bias+2*s_p_num]=sampled_z; // take the first point as a sampled point
        for (int64_t i=1;i<s_p_num;i++){ // loop for sampling
            cur_max_dist=-1.0;
            for (int64_t j=0;j<o_p_num;j++){ // loop for brute force search
                index=o_index_bias+j*3;
                cur_x=original[index]; cur_y=original[index+1]; cur_z=original[index+2];

                // index=o_index_bias+j;
                // cur_x=original[index]; 
                // cur_y=original[index+o_p_num]; 
                // cur_z=original[index+2*o_p_num];
                dist_x=cur_x-sampled_x; dist_y=cur_y-sampled_y; dist_z=cur_z-sampled_z;
                dist=dist_x*dist_x+dist_y*dist_y+dist_z*dist_z;
                if (dist<max_dist[j]) max_dist[j]=dist; //update the minimal distance to sampled points for current point  
                if (max_dist[j]>cur_max_dist) {cur_max_dist=max_dist[j]; selected_id=j;} //update the maximal distance and point id for current sampling
            }
            index=o_index_bias+selected_id*3;
            sampled_x=original[index]; sampled_y=original[index+1]; sampled_z=original[index+2];

            // index=s_index_bias+i*3;
            // sampled[index]=sampled_x; sampled[index+1]=sampled_y; sampled[index+2]=sampled_z;

            // index=o_index_bias+selected_id;
            // sampled_x=original[index]; 
            // sampled_y=original[index+o_p_num]; 
            // sampled_z=original[index+2*o_p_num];

            index=s_index_bias+i;
            sampled[index]=sampled_x; 
            sampled[index+s_p_num]=sampled_y; 
            sampled[index+2*s_p_num]=sampled_z;// take the point with maximal distance as a sampled point
        }
    }
    delete []max_dist;

    //the following test the behavior of ONNX Runtime and ONNX model data transmission
    std::cout<<"test behavior: "<<tt<<std::endl; 
    const OrtValue *test_num=ort_.KernelContext_GetInput(context,2);
    const double *test_num_value=reinterpret_cast<const double*>(ort_.GetTensorData<double>(test_num));
    std::cout<<test_num_value[0]<<std::endl; // test to get a scalar
};
#endif