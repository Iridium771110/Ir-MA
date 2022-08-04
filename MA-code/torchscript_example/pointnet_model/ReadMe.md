This category includes an example to test the torchscript model of PointNet.
The original PointNet project is from https://github.com/fxia22/pointnet.pytorch 
Clone this project and install it to support the model loading and verification

In folder data, the test data and a TorchScript model and a PyTorch model of PointNet for classification are stored.

Modify the path to model and the path to data, run command

python to_script.py

to generate the TorchScript model and 

python to_onnx.py

to generate the ONNX model and make the test combined with a TorchScript custom operator FPS

Modify the CMakeLists and the data path and the model path to test the model.

mkdir build
cd build
cmake ..
make
./try_script ../data/cls_script_model_pnt.pt