#include<torch/script.h>
#include<iostream>
#include<fstream>
#include<memory>
#include<vector>
/*
This code is for test of torchscript model of PointNet on the CPU
*/

int cmp(torch::Tensor output, long *target, int start_index, int num){
//count the number of predictions that match the ground truth.
  int max_index;
  float max_p;
  int equal=0;
  float p;

  output=output.to(at::kCPU);
  auto outdata=output.accessor<float,2>(); //get the data table

  for (int i=0;i<num;i++){
    max_p=-10000000.0;
    for(int j=0;j<16;j++) {
      p=outdata[i][j];
      if (max_p<p) {max_p=p;max_index=j;}
    }
    if (max_index==target[start_index+i]) equal++;
  }
  return equal;
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
      std::cerr << "usage: example-app <path-to-exported-script-module>\n";
      return -1;
    }

    torch::jit::script::Module module;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      return -1;
    }
    std::cout << "load module ok\n";

    std::ifstream datain("../data/test_data.bin",std::ios::binary);
    std::ifstream targetin("../data/test_target.bin",std::ios::binary);
    float *data= new float[2874*3*2500];
    float single_data;
    int index=0;
    long *target= new long[2874]; // the number of test samples should be adapted to the test set
    int t_index=0;
    long single_target;
    while (datain.read((char*)&single_data,sizeof(float))){ // read the point data
        data[index*3]=single_data;
        datain.read((char*)&single_data,sizeof(float));
        data[index*3+1]=single_data;
        datain.read((char*)&single_data,sizeof(float));
        data[index*3+2]=single_data;
        index++;
    }
    datain.close();
    while (targetin.read((char*)&single_target,sizeof(long))){// read the target label
      target[t_index]=single_target;
      t_index++;
    }
    targetin.close();
    std::cout<<index<<' '<<"finish reading "<<t_index<<std::endl;

    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input_data;
    at::Tensor output;
    inputs.clear();

    int batch_size=16;
    int i,bias;
    int matched=0;
    for (i=0;i<t_index/batch_size;i++){ // inference and check the label
      bias=7500*batch_size*i;
      input_data=torch::from_blob(data+bias,{batch_size,3,2500});//generate input tensor
      input_data=input_data.to(at::kCUDA);
      inputs.push_back(input_data);
      output=module.forward(inputs).toTensor();//do inference
      matched+=cmp(output,target,i*batch_size,batch_size);//count the number of correct results
      inputs.clear();
    }
    std::cout<<"well"<<std::endl;
    if (i*batch_size<t_index){// inference and check the result for the remaining samples (case number of samples is not divisible by batch size)
      bias=7500*batch_size*i;
      input_data=torch::from_blob(data+bias,{t_index-batch_size*i,3,2500});
      input_data=input_data.to(at::kCUDA);
      inputs.push_back(input_data);
      output=module.forward(inputs).toTensor();
      matched+=cmp(output,target,i*batch_size,t_index-batch_size*i);
    }

    //std::cout<<input_data.device()<<std::endl;
    //std::cout<<output.slice(0,0,16)<<std::endl<<output.dim()<<std::endl;
    //std::cout<<std::endl; //check the output's state

    std::cout<<matched<<' '<<float(matched)/float(t_index)<<std::endl;

    delete []data;
    delete []target;
    return 0;
}