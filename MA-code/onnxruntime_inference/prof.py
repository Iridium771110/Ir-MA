import json
import os
# This file is used to handle the profiling file in json form
# It includes some specify locations for extraction of desired information

root="/home/dong/WS/test/onnx_ops/naive-test-prof-p" # please modify the path to profiling file
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
                if not line:
                    break
                if line.__len__()==2:
                    continue

                if line[-2]==',': # adapt the line for json.loads()
                    line=line[:-2]
                else:
                    line=line[:-1]
                js=json.loads(line)
                node_name=js["name"] # extract the name of item
                if node_name[-4:]!="time": # only count the time item
                    continue
                if node_name in time_dict.keys() : # extract the duration and count the time cost
                    time_dict[node_name]+=js["dur"]
                else:
                    time_dict[node_name]=js["dur"]
        jsf.close()
        out_file=file_name
        with open(out_file,'w') as out_jsf :
            json.dump(time_dict,out_jsf,indent=2)
        out_jsf.close()

