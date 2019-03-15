import os
d_path='./flower_photos/daisy'
r_path='./flower_photos/roses'
d=os.listdir(d_path)
r=os.listdir(r_path)
train_file = './train.txt'
val_file = './val.txt'
k=5
d_list=[_ for _ in d]
r_list=[_ for _ in r]
length=len(d_list)
length_t=int(length/k*(k-1))
#write the train data
with open(train_file,"w") as f:
    for file in d_list[:length_t]:
        f.write(d_path+'/'+file+' 0'+'\n')
    for file in r_list[:length_t]:
        f.write(r_path+'/'+file +' 1'+'\n')
#write the validation data
with open(val_file,"w") as f:
    for file in d_list[length_t:]:
        f.write(d_path+'/'+file+' 0'+'\n')
    for file in r_list[length_t:]:
        f.write(r_path+'/'+file +' 1'+ '\n')
        #5679288570_b4c52e76d5