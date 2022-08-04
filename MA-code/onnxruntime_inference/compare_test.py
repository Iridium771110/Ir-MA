import os
import json
# This file tries to execute multiple tests with different batch sizes and write the test results into a json file

b_size=['1','2','4','8']

run_object="build/onnx_ssg " # please modify the path to execution file
model_path=" data/script_model.onnx" # please modify the path to ONNX model
data_path=" data/test_data.bin" # please modify the path to test data
label_path=" data/test_label.bin" # please modify the path to target label
test_len=16
time_b=[]
acc_b=[]

# execute multiple tests and store the results
for b in b_size:
    test_cmd=run_object+b+model_path+data_path+label_path
    for i in range(0,test_len):
        re=os.popen(test_cmd)
        time=re.readlines()
        l=time.__len__()
        time_b.append(time[l-1])
        acc_b.append(time[l-3][17:])

# write the test results into a json file
item=0
fw=open("test.json",'w')
for b in b_size:
    time_total=0.0
    time_ave=0.0
    print("**************************")
    print(b,":")
    for i in range(1,test_len):
        item+=1
        print("time cost: ",time_b[item],"acc: ",acc_b[item])
        test_dict={"batch_size":b,"time cost":time_b[item],"acc":acc_b[item]}
        fw.write(json.dumps(test_dict))
        fw.write('\n')
        time_total+=float(time_b[item])
    time_ave=time_total/float(test_len-1)
    item+=1
    print("time ave.: ", time_ave)
    print("___________________________")
fw.close()