import os
import json

b_size=['1','2','4','8']
# a="sss sss"
# for b in b_size:
#     c=a+b
#     print(c)

# a="134.2984"
# c=float(a)
# d=c+2
# e=(c+d)/2
# print(c,d,e)

run_object="build/onnx_ssg "
model_path=" data/script_model.onnx"
data_path=" data/test_data.bin"
label_path=" data/test_label.bin"
test_len=12
time_b=[]
acc_b=[]

# test_cmd=run_object+b_size[0]+model_path+data_path+label_path
# re=os.popen(test_cmd)
# res=re.readlines()
# l=res.__len__()
# print(l)
# print(res[l-1],res[l-3][18:])
# test_dict={"time cost":res[l-1],"acc":res[l-3][18:]}
# fw=open("test.json",'w')
# fw.write(json.dumps(test_dict))
# fw.write('\n')
# fw.write(json.dumps(test_dict))
# fw.write('\n')
# fw.write(json.dumps(test_dict))
# fw.close()

for b in b_size:
    test_cmd=run_object+b+model_path+data_path+label_path
    #test_cmd="build/onnx_ssg 1 data/script_model.onnx data/test_data.bin data/test_label.bin"
    for i in range(0,test_len):
        re=os.popen(test_cmd)
        time=re.readlines()
        l=time.__len__()
        time_b.append(time[l-1])
        acc_b.append(time[l-3][17:])

item=0
fw=open("test-runtime-oct-fcs-p.json",'w')
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