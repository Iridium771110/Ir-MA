This category converts the trained model from PyTorch to ONNX
The trained model is trained by the original project: https://github.com/erikwijmans/Pointnet2_PyTorch
The codes from the original project are modified here for the happiness of convertor
The model has been rewriten and some unnecessary modules are deleted since only SSG is focused
The path to data (ModelNet40) should be modified
The model can also be trained using the modified codes here

Once the shared library is installed, and the path to lib is modified, run command:

python pnt2_tr.py

to train the PointNet++ model for SSG
Then run command:

python convert.py

to generate the TorchScript model and the ONNX model
The ONNX model script_model.onnx will be used in inference test