import json
import os

cost_dict={
    "115_nchwc_kernel_time":0.0,
    "118_nchwc_kernel_time":0.0,
    "132_nchwc_kernel_time":0.0,
    "135_nchwc_kernel_time":0.0,
    "89_nchwc_kernel_time":0.0,
    "92_nchwc_kernel_time":0.0,
    "95_nchwc_kernel_time":0.0,
    "Add_87_kernel_time":0.0,
    "BatchNormalization_81_kernel_time":0.0,
    "BatchNormalization_84_kernel_time":0.0,
    "Concat_28_kernel_time":0.0,
    "Concat_48_kernel_time":0.0,
    "Concat_62_kernel_time":0.0,
    "Concat_70_kernel_time":0.0,
    "ConcatFromSequence_37_kernel_time":0.0,
    "ConcatFromSequence_57_kernel_time":0.0,
    "Equal_75_kernel_time":0.0,
    "fused Conv_49_kernel_time":0.0
}

root="/home/dong/WS/test/onnx_ops/naive-test-prof-p"
file_names=os.listdir(root)

for file_name in file_names :
    time_dict={}
    file_path=root+'/'+file_name
    if os.path.isfile(file_path):
        print(file_path)
        with open(file_path,encoding='utf-8') as jsf :
            i=0
            
            while (1):
                i+=1
                line=jsf.readline()
                #print(line[:-2])
                # if i>=229526:
                #     print(line[-2])
                if not line:
                    break
                #print(line.__len__())
                if line.__len__()==2:
                    continue

                if line[-2]==',':
                    line=line[:-2]
                else:
                    line=line[:-1]
                #line='r'+line
                js=json.loads(line)
                #print(js)
                #print(i)
                node_name=js["name"]
                #print(node_name[-4:])
                if node_name[-4:]!="time":
                    continue
                if node_name in time_dict.keys() :
                    time_dict[node_name]+=js["dur"]
                else:
                    time_dict[node_name]=js["dur"]
        jsf.close()
        out_file=file_name
        with open(out_file,'w') as out_jsf :
            json.dump(time_dict,out_jsf,indent=2)
        out_jsf.close()

