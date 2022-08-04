This category is copied from the original project and modified for the inference test in PyTorch
For the benchmark test, the custom operators' behavior in naive version are implemented in cpp files under folder _ext_src/src

run command:

python setup.py install

to install the custom operators. To test the cost without custom operators, please comment out the naive implementation and uncomment the placeholder. And reinstall these operators.