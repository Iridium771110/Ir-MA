#include "sampling.h"
#include "gather_points.h"
#include "grouping.h"
#include "ball_query.h"
#include <fstream>

bool test_pnt2( Ort::Env &env,
                const char* model_path,
                const char* points_path,
                const char* label_path,
                const char* input_name,
                const char* output_name,
                const int points_num,
                OrtCustomOpDomain* custom_op_domain_ptr){
    Ort::SessionOptions session_options;
    std::cout<<"pnt2 test with default provider"<<std::endl;

    if (custom_op_domain_ptr) session_options.Add(custom_op_domain_ptr);
    else std::cout<<"wtf"<<std::endl;
    std::cout<<"wtf1"<<std::endl;
    Ort::Session session(env,model_path,session_options);
    std::cout<<"wtf2"<<std::endl;
    auto memory_info=Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;
    std::vector<int64_t> input_dim(3);
    //input_dim[0]=1;input_dim[1]=points_num;input_dim[2]=6;
    input_dim[0]=1;input_dim[1]=5;input_dim[2]=5;
    input_names.emplace_back(input_name);
    std::vector<float> points={1.2987, -0.9477, 0.6843, 0.3944, 2.0187,
                                0.9125, 0.9375, 0.2431, -0.3821, 0.0988,
                                2.4182, 1.7426, -0.9073, 1.1238, -1.2532,
                                1.6646, -0.2011, -1.0471, 0.9630, 0.4516,
                                0.8094, 0.7476, -0.4819, 1.0698, -2.2114};
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>( memory_info,
                                                                const_cast<float*>(points.data()),
                                                                points.size(),
                                                                input_dim.data(),
                                                                input_dim.size()));

    std::vector<const char*> output_names;
    output_names.emplace_back(output_name);

    std::vector<Ort::Value> ort_output;
    ort_output=session.Run( Ort::RunOptions{nullptr},
                            input_names.data(),
                            input_tensors.data(),
                            input_tensors.size(),
                            output_names.data(),1);
    
    for (int64_t i=0;i<ort_output.size();i++){
        int* f=ort_output[i].GetTensorMutableData<int>();
        auto type_info=ort_output[i].GetTensorTypeAndShapeInfo();
        for (int64_t j=0;j<type_info.GetElementCount();j++){
            if (j%5==0) std::cout<<std::endl;
            std::cout<<f[j]<<' ';
        }
    }
    std::cout<<std::endl;
    // std::ifstream points_input(points_path,std::ios::binary);
    // std::ifstream labels_input(label_path,std::ios::binary);
    // int label,output_label;
    // int64_t output_num,total_len,correct,total;
    // float point_axis_value,max_f;
    // std::vector<float> points(points_num*6);
    // correct=0; total=0;
    // while(labels_input.read((char*)&label,sizeof(int))){
    //     total++;
    //     for (int i=0;i<points_num*6;i++){
    //         points_input.read((char*)&point_axis_value,sizeof(float));
    //         points[i]=point_axis_value;
    //     }
    
    //     //input_tensors[0] = input_tensor;
    //     input_tensors.clear();
    //     // Ort::Value input_tensor=Ort::Value::CreateTensor<float>(memory_info,
    //     //                                                         const_cast<float*>(points.data()),
    //     //                                                         points.size(),
    //     //                                                         input_dim.data(),
    //     //                                                         input_dim.size());
    //     // input_tensors.emplace_back(input_tensor);
    //     input_tensors.emplace_back(Ort::Value::CreateTensor<float>( memory_info,
    //                                                                 const_cast<float*>(points.data()),
    //                                                                 points.size(),
    //                                                                 input_dim.data(),
    //                                                                 input_dim.size()));
    //     std::vector<Ort::Value> ort_output;
    //     ort_output=session.Run( Ort::RunOptions{nullptr},
    //                             input_names.data(),
    //                             input_tensors.data(),
    //                             input_tensors.size(),
    //                             output_names.data(),1);

    //     output_num=ort_output.size();
    //     for (int64_t i=0;i<output_num;i++){
    //         float* f=ort_output[i].GetTensorMutableData<float>();
    //         auto type_info=ort_output[i].GetTensorTypeAndShapeInfo();
    //         total_len=type_info.GetElementCount();

    //         max_f=-1e10;
    //         for (int64_t j=0;j<total_len;j++){
    //             if (f[j]>max_f){
    //                 max_f=f[j];
    //                 output_label=j;
    //             }
    //         }
    //         if (output_label==label) correct++;
    //     }
    // }
    // points_input.close();
    // labels_input.close();

    // std::cout<<"correct num: "<<correct<<", total num: "<<total<<std::endl;
    // std::cout<<"pseudo test acc: "<<float(correct)/float(total)<<std::endl;
    return true;
}
int main(int argc, char** argv){
    if (argc<2) throw "please give onnx model path";
    if (argc<3) throw "please give points data path";
    if (argc<4) throw "please give labels path";
    const char* model_path=argv[1];
    const char* data_path=argv[2];
    const char* label_path=argv[3];
    Ort::Env env_=Ort::Env(ORT_LOGGING_LEVEL_INFO,"Default");
    
    SamplingCustomOp onnx_sampling;
    GatherPointsCustomOp onnx_gather_points;
    GroupingCustomOp onnx_grouping;
    BallQueryCustomOp onnx_ball_query;
    Ort::CustomOpDomain custom_op_domain("onnx_pnt2_ops");
    custom_op_domain.Add(&onnx_sampling);
    custom_op_domain.Add(&onnx_gather_points);
    custom_op_domain.Add(&onnx_grouping);
    custom_op_domain.Add(&onnx_ball_query);

    std::cout<<test_pnt2(env_,model_path,data_path,label_path,"points","results",4096,custom_op_domain)<<std::endl;
    
    return 0;
}