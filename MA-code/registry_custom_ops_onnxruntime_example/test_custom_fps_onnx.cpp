#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "test_custom_fps_onnx.h"
/*
This code is the test file using ONNX Runtime for a ONNX model
*/

template <typename T>
bool test_fps_op(Ort::Env&env, 
                T model_uri,
                const Input1& inputs1,
                const Input2& inputs2,
                const char* output_name,
                const std::vector<int64_t>& expected_dims_y,
                const std::vector<float>& expected_values_y,
                OrtCustomOpDomain* custom_op_domain_ptr){
    // inference using the ONNX model with custom operator
    // Parameters: onnxruntime environment, path to ONNX model, first input, second input, name of output, expected output dimension,
    //             expected output value, onnxruntime custom operators domain
    
    Ort::SessionOptions session_options;
    std::cout<<"simple test with default provider"<<std::endl;

    if (custom_op_domain_ptr) {session_options.Add(custom_op_domain_ptr);}
    Ort::Session session(env,model_uri,session_options);// initialization and set the onnxruntime session

    auto memory_info=Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;

    input_names.emplace_back(inputs1.name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>( memory_info,
                                                                const_cast<float*>(inputs1.values.data()),
                                                                inputs1.values.size(),
                                                                inputs1.dims.data(),
                                                                inputs1.dims.size()));// set the input 
    input_names.emplace_back(inputs2.name);
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                const_cast<int64_t*>(inputs2.values.data()),
                                                                inputs2.values.size(),
                                                                inputs2.dims.data(),
                                                                inputs2.dims.size()));// set the input
    std::vector<Ort::Value> ort_outputs;
    ort_outputs=session.Run(Ort::RunOptions{nullptr},input_names.data(),input_tensors.data(),input_tensors.size(),&output_name,1);// do inference
    Ort::Value expected_output_tensor{nullptr};
    expected_output_tensor=Ort::Value::CreateTensor<float>( memory_info,
                                                            const_cast<float*>(expected_values_y.data()),
                                                            expected_values_y.size(),
                                                            expected_dims_y.data(),
                                                            expected_dims_y.size());// set the example output
    //assert(ort_outputs.size()==1);

    auto type_info=expected_output_tensor.GetTensorTypeAndShapeInfo();
    //assert(type_info.GetShape()==expected_dims_y);
    size_t total_len=type_info.GetElementCount();
    //assert(expected_values_y.size()==total_len);
    
    int64_t output_num=ort_outputs.size();
    std::cout<<output_num<<std::endl;
    for (int64_t i=0;i<output_num;i++){
        // print the result to check the model behavior
        float *f=ort_outputs[i].GetTensorMutableData<float>();// get the output data ptr
        type_info=ort_outputs[i].GetTensorTypeAndShapeInfo();// get output tensor information
        total_len=type_info.GetElementCount();// get the element number of output tensor
        for (int64_t j=0;j<total_len;j++){
            if (j%5==0) std::cout<<std::endl;
            std::cout<<f[j]<<' ';
        }
        std::cout<<std::endl;
    }

    return true;
};

int main(int argc, char **argv){
    FpsCustomOp custom_op;
    Ort::CustomOpDomain custom_op_domain("custom_ops_node");
    custom_op_domain.Add(&custom_op); // register the custom operator with ONNX Runtime

    if (argc<2) return 0;
    const char *model_path=argv[1];
    Ort::Env env_=Ort::Env(ORT_LOGGING_LEVEL_INFO,"Default");

    std::vector<int64_t> expected_dims_y={1,3,5};
    std::vector<float> expected_values_y={  1.0f,1.0f,1.0f,1.0f,1.0f,
                                            1.0f,1.0f,1.0f,1.0f,1.0f,
                                            1.0f,1.0f,1.0f,1.0f,1.0f}; // used for comparison with the output, but meaningless here

    Input1 inputs1;
    inputs1.name="original";
    inputs1.dims={1,3,5};
    inputs1.values={1.1f,1.2f,1.3f,
                    2.4f,0.4f,2.1f,
                    1.4f,3.5f,0.4f,
                    2.1f,3.4f,4.2f,
                    0.5f,0.9f,1.5f};
    Input2 inputs2;
    inputs2.name="to_sample";
    inputs2.dims={1,3,5};
    inputs2.values={0,0,0,0,0,
                    0,0,0,0,0,
                    0,0,0,0,0};// set the test inputs sample
    std::cout<<test_fps_op(env_,model_path,inputs1,inputs2,"sampled",expected_dims_y,expected_values_y,custom_op_domain)<<std::endl;

    return 0;
}