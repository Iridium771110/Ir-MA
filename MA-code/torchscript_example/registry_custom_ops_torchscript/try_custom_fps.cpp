#include<torch/script.h>
#include<vector>
#include<math.h>
/*
This code is an example to register a custom operator with TorchScript
It is just one way to realize the registration
The furthest point sampling operator is taken as the example
*/

torch::Tensor fps(torch::Tensor input, torch::Tensor output, double a){
    //custom operator that performs the furthest point sampling algorithm
    
    // get the information of inputs and set the parameters
    float *original=input.data_ptr<float>();
    //int o_point_num=int(o_point_n.data_ptr<int>()[0]);
    int o_point_num=input.size(2);
    //int s_point_num=int(s_point_n.data_ptr<int>()[0]);
    int s_point_num=output.size(2);
    //int batch_size=int(b_size.data_ptr<int>()[0]);
    int batch_size=input.size(0);
    float *sampled=new float[3*s_point_num*batch_size];
    //float *sampled=output.data_ptr<float>();
    float *max_dist=new float[o_point_num];
    float dist,cur_max_dist;
    float cur_x,cur_y,cur_z,sampled_x,sampled_y,sampled_z;
    int selected_id=0,index;
    int s_index_bias,o_index_bias;

    for (int batch_num=0;batch_num<batch_size;batch_num++){ // loop for batch size
        s_index_bias=batch_num*s_point_num*3;
        o_index_bias=batch_num*o_point_num*3;
        for (int i=0;i<o_point_num;i++) max_dist[i]=1e10; //initialize
        sampled_x=original[o_index_bias];
        sampled_y=original[o_index_bias+1]; 
        sampled_z=original[o_index_bias+2];

        // sampled[s_index_bias]=sampled_x;
        // sampled[s_index_bias+1]=sampled_y;
        // sampled[s_index_bias+2]=sampled_z;

        // sampled_x=original[o_index_bias];
        // sampled_y=original[o_index_bias+o_point_num]; 
        // sampled_z=original[o_index_bias+2*o_point_num];

        sampled[s_index_bias]=sampled_x;
        sampled[s_index_bias+s_point_num]=sampled_y;
        sampled[s_index_bias+2*s_point_num]=sampled_z; //take the first point as a result
        for (int i=1;i<s_point_num;i++){ // loop for sampling
            cur_max_dist=-1.0;
            for (int j=0;j<o_point_num;j++){ // loop for brute force search
                index=o_index_bias+j*3;
                cur_x=original[index]; cur_y=original[index+1]; cur_z=original[index+2];

                // index=o_index_bias+j;
                // cur_x=original[index]; 
                // cur_y=original[index+o_point_num]; 
                // cur_z=original[index+2*o_point_num];
                dist=pow(cur_x-sampled_x,2)+pow(cur_y-sampled_y,2)+pow(cur_z-sampled_z,2);
                if (dist<max_dist[j]) max_dist[j]=dist; //update the minimal distance to sampled points for current point 
                if (max_dist[j]>cur_max_dist) {cur_max_dist=max_dist[j]; selected_id=j;} //update the maximal distance and point id for current sampling
            }
            index=o_index_bias+selected_id*3;
            sampled_x=original[index]; sampled_y=original[index+1]; sampled_z=original[index+2];

            // index=s_index_bias+i*3;
            // sampled[index]=sampled_x; sampled[index+1]=sampled_y; sampled[index+2]=sampled_z;

            // index=o_index_bias+selected_id;
            // sampled_x=original[index]; 
            // sampled_y=original[index+o_point_num]; 
            // sampled_z=original[index+2*o_point_num];

            index=s_index_bias+i;
            sampled[index]=sampled_x; 
            sampled[index+s_point_num]=sampled_y; 
            sampled[index+2*s_point_num]=sampled_z; // take the point with maximal distance as a sampled point
        }
    }

    torch::Tensor result=torch::from_blob(sampled,{batch_size,3,s_point_num}); //generate the result tensor
    delete []max_dist;
    return result;
}

static auto registry=torch::RegisterOperators("custom_ops::test_fps",&fps); //register the operator with TorchScript (just one of feasible ways)