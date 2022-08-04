import torch
import time
from net import pnt2_cls_ssg
import numpy as np

# no need to retrain the model, just test the inference speed here

def test(BATCH_SIZE):
    #NUM_BATCH = 100
    #BATCH_SIZE = 8

#   set the configuration of inference test
    NUM_POINTS = 4096
    stat = 0    
    net = pnt2_cls_ssg().cpu()
    net.load_state_dict(torch.load('../pnt2_cls_ssg.pth')) # please modify the path to model
    net.eval()
    
    all_data=np.fromfile("../test_data.bin",dtype=np.float32) # please modify the path to data
    all_label=np.fromfile("../test_label.bin",dtype=np.int64) # please modify the path to data
    num_data=all_label.shape[0]
    one_batch=all_data.shape[0]//num_data
    num_batch=num_data//BATCH_SIZE
    print(one_batch)

#   load the data and label
    data_list=[]
    label_list=[]
    idx_0=0
    idx_1=0
    idx_2=0
    idx_3=0
    for i in range(0,num_batch):
        idx_0=i*one_batch*BATCH_SIZE
        idx_1=idx_0+one_batch*BATCH_SIZE
        idx_2=i*BATCH_SIZE
        idx_3=idx_2+BATCH_SIZE
        data_list.append(all_data[idx_0:idx_1].copy())
        label_list.append(all_label[idx_2:idx_3].copy())

    idx_0=idx_1
    idx_2=idx_3
    data_list.append(all_data[idx_0:].copy())
    label_list.append(all_label[idx_2:].copy())
    print("data loaded")

#   test the inference in python
    with torch.no_grad():
        num_correct=0
        for data,label in zip(data_list,label_list):
            b_size=label.shape[0]
            data=data.reshape(b_size,NUM_POINTS,6)
            
            x=torch.from_numpy(data).cpu()
            labels=torch.from_numpy(label)

            begin = time.time()
            y = net(x)    # do not measure the data generating time
            stat += time.time() - begin

            re_labels=torch.max(y,dim=-1)[1]
            num_correct+=torch.sum(re_labels==labels)

    print(num_correct)
    print("time all: ", stat)
    print("time per data: ", stat / num_data)

    return stat
    

if __name__ == "__main__":

#   test the inference in python and store the results as a list
    import json
    b_size=[1,2,4,8]
    test_len=8
    time_list=[]
    for b in b_size:
        for _ in range(0,test_len):
            time_list.append(np.copy(test(b)))

#   write the result in the json file
    item=0
    fw=open("test.json",'w')
    for b in b_size:
        time_total=0.0
        time_ave=0.0
        print("**************************")
        print(b,":")
        for i in range(0,test_len):
            print("time cost: ",time_list[item])
            test_dict={"batch_size":b,"time cost":float(time_list[item])}
            fw.write(json.dumps(test_dict))
            fw.write('\n')
            time_total+=float(time_list[item])
            item+=1
        time_ave=time_total/float(test_len)

        print("time ave.: ", time_ave)
        print("___________________________")
    fw.close()