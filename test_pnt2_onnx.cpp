#include "sampling.h"
#include "gather_points.h"
#include "grouping.h"
#include "ball_query_tree.h"
#include <fstream>

#include <sys/time.h>
#include<iomanip>

bool test_pnt2( Ort::Env &env,
                const char* model_path,
                const char* points_path,
                const char* label_path,
                const char* input_name,
                const char* output_name,
                const int points_num,
                const int64_t b_size,
                OrtCustomOpDomain* custom_op_domain_ptr){
    Ort::SessionOptions session_options;
    std::cout<<"pnt2 test with default provider"<<std::endl;

    if (custom_op_domain_ptr) session_options.Add(custom_op_domain_ptr);
    else std::cout<<"wtf"<<std::endl;
    //std::cout<<"wtf1"<<std::endl;
    session_options.SetLogSeverityLevel(2);
    // char profile_name[3]; profile_name[0]='b'; profile_name[1]='=';
    // profile_name[2]=b_size+'0';
    // session_options.EnableProfiling(profile_name);
    Ort::Session session(env,model_path,session_options);
    //std::cout<<"wtf2"<<std::endl;
    auto memory_info=Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;
    //int64_t b_size=1;
    std::vector<int64_t> input_dim(3);
    input_dim[0]=b_size;input_dim[1]=points_num;input_dim[2]=6;
    input_names.emplace_back(input_name);

    std::vector<const char*> output_names;
    output_names.emplace_back(output_name);
 
    std::ifstream points_input(points_path,std::ios::binary);
    std::ifstream labels_input(label_path,std::ios::binary);
    int64_t label[b_size];
    int64_t output_label,input_label;
    int64_t output_num,total_len,correct,total;
    float point_axis_value,max_f;
    std::vector<float> points(points_num*6*b_size);
    correct=0; total=0;
    while(labels_input.read((char*)&label[0],sizeof(int64_t))){
        total_len=1;
        while(total_len<b_size){
            if (labels_input.read((char*)&input_label,sizeof(int64_t))) {label[total_len]=input_label;total_len++;}
            else break;
        }
        total+=total_len;
        //std::cout<<total<<' '<<label[0]<<std::endl;
        for (int i=0;i<points_num*6*total_len;i++){
            points_input.read((char*)&point_axis_value,sizeof(float));
            points[i]=point_axis_value;
        }
    
        //input_tensors[0] = input_tensor;
        input_tensors.clear();
        input_dim[0]=total_len;
        // Ort::Value input_tensor=Ort::Value::CreateTensor<float>(memory_info,
        //                                                         const_cast<float*>(points.data()),
        //                                                         points.size(),
        //                                                         input_dim.data(),
        //                                                         input_dim.size());
        // input_tensors.emplace_back(input_tensor);
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>( memory_info,
                                                                    const_cast<float*>(points.data()),
                                                                    //points.size(),
                                                                    points_num*6*total_len,
                                                                    input_dim.data(),
                                                                    input_dim.size()));
        std::vector<Ort::Value> ort_output;
        ort_output=session.Run( Ort::RunOptions{nullptr},
                                input_names.data(),
                                input_tensors.data(),
                                input_tensors.size(),
                                output_names.data(),1);

        //output_num=ort_output.size();
        //std::cout<<output_num<<std::endl;
        float* f=ort_output[0].GetTensorMutableData<float>();
        auto type_info=ort_output[0].GetTensorTypeAndShapeInfo();
        int64_t output_dim_num=type_info.GetDimensionsCount();
        int64_t output_dim[output_dim_num];
        type_info.GetDimensions(output_dim,output_dim_num);
        //std::cout<<output_b_size[0]<<std::endl;
        if (output_dim[0]!=total_len) throw "batch size problem";
        total_len=output_dim[1];

        for (int64_t i=0;i<output_dim[0];i++){
            max_f=-1e10;
            for (int64_t j=0;j<total_len;j++){
                if (f[j+i*total_len]>max_f){
                    max_f=f[j+i*total_len];
                    output_label=j;
                }
            }
            if (output_label==label[i]) correct++;
        }
        //break;
    }
    points_input.close();
    labels_input.close();

    std::cout<<"correct num: "<<correct<<", total num: "<<total<<std::endl;
    std::cout<<"pseudo test acc: "<<float(correct)/float(total)<<std::endl;
    return true;
}
int main(int argc, char** argv){
    if (argc<2) throw "please give batch size";
    if (argc<3) throw "please give onnx model path";
    if (argc<4) throw "please give points data path";
    if (argc<5) throw "please give labels path";
    const char* b_s=argv[1];
    int64_t batch_size=0;
    int item=0;
    while(b_s[item]!='\0'){
        batch_size=batch_size*10+b_s[item]-'0';
        item++;
    }
    const char* model_path=argv[2];
    const char* data_path=argv[3];
    const char* label_path=argv[4];
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

    //max_dist= new float[4096];

    struct timeval t1,t2;
    long sec,usec;
    gettimeofday(&t1,NULL);

    std::cout<<test_pnt2(env_,model_path,data_path,label_path,"points","results",4096,batch_size,custom_op_domain)<<std::endl;

    gettimeofday(&t2,NULL);
    usec=t2.tv_usec-t1.tv_usec;
    sec=t2.tv_sec-t1.tv_sec;
    if (usec<0) {
        sec-=1;
        usec=-usec;
    }
    std::cout<<sec<<'.';
    std::cout<<std::setw(6)<<std::setfill('0')<<usec<<std::endl;
    
    //delete []max_dist;
    //max_dist=nullptr;

    return 0;
}