This category makes the inference test with ONNX model and ONNX Runtime on the CPU
Folder data includes the test data and the test ONNX model
Folder kd_tree and oct_tree include the implementation of k-d tree and octree structures

The ball query operator has 3 versions:
ball_query.h for brute force search
ball_query_kd.h for k-d tree-based search
ball_query_tree.h for octree-based search
Change the include head file in test_pnt2_onnx.cpp to select the version

The gather points operator has 2 versions:
original access sequence 
an access sequence with higher cache hit ratio
But in practice, the cost of this operator is very low, only the original version is tested

The grouping operator has 3 versions:
original access sequence
an access sequence with higher cache hit ratio
multi-threading using OpenMP
Comment out and uncomment the corresponding parts in file grouping.h to select the version

run command:

mkdir build
cd build
cmake ..
make 
./onnx_ssg "batch size" "path to model" "path to point cloud" "path to target label"

to test the inference with ONNX Runtime on the CPU for a single time
Once the executable file is generated, run command:

python compare_test.py 

to test the inference for multiple times and batch sizes
The results will be writen into a json file