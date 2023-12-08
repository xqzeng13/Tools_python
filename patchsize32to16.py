import pandas as pd
import glob
import os
import numpy as np
import openpyxl
import torch

if __name__ == '__main__':

    # model="write"
    # model="read"
    # csvname=r'E:\BraTS_64x64x64_samples\trian_val.csv'
    normlization= False
    Exsit_training_path = r'G:\lung\nii\train_val_ord\\'
#   Exsit_val_path = r'D:\Work\Datasets\samples\debug_loss\val\\'
#   Exsit_test_path = r'D:\Work\Datasets\samples\debug_loss\test\\'
    datanameList = sorted(glob.glob(os.path.join(Exsit_training_path, '*0.npy')))
    labelnameList = sorted(glob.glob(os.path.join(Exsit_training_path, '*seg.npy')))
    assert len(datanameList)==len(labelnameList)
    datanameList.sort()
    labelnameList.sort()
    for i in range(len(labelnameList)):
        # num = labelnameList[i].split('seg.npy')[0].split('__')[1]
        path=r'G:\lung\nii\32x32x32\\'
        name=str(labelnameList[i]).split('train_val_ord\\')[-1].split('__')[0]#'train_order(0, 0, 0)'
        data_arr = np.load(datanameList[i])
        label_arr = np.load(labelnameList[i])
        ##归一化
        if normlization:####64x64x32 没有做标准化；32x32x32做了
            data_tensor=torch.tensor(data_arr)
            MEAN, STD = data_tensor.mean(), data_tensor.std()
            # MAX, MIN = data_tensor.max(), data_tensor.min()
            img_tensor = (data_tensor.clone() - MEAN) / STD##均值方差归一化
            data_arr=np.array(img_tensor)

        else:
            data_arr=data_arr


        ##todo crop  01
        data01=data_arr[:32,:32,:]
        data02=data_arr[:32,32:64,:]
        data03=data_arr[32:64,:32,:]
        data04=data_arr[32:64,32:64,:]
        name1=path+name+"01"+'__'+str(datanameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        name2=path+name+"02"+'__'+str(datanameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        name3=path+name+"03"+'__'+str(datanameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        name4=path+name+"04"+'__'+str(datanameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        np.save(name1,data01)
        np.save(name2,data02)
        np.save(name3,data03)
        np.save(name4,data04)

        label01=label_arr[:32,:32,:]
        label02=label_arr[:32,32:64,:]
        label03=label_arr[32:64,:32,:]
        label04=label_arr[32:64,32:64,:]
        name11=path+name+"01"+'__'+str(labelnameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        name22=path+name+"02"+'__'+str(labelnameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        name33=path+name+"03"+'__'+str(labelnameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        name44=path+name+"04"+'__'+str(labelnameList[i]).split('train_val_ord\\')[-1].split('__')[1]
        np.save(name11,label01)
        np.save(name22,label02)
        np.save(name33,label03)
        np.save(name44,label04)
        print('\r[ %d / %d]' % (i, len(labelnameList)), end='')




    '''
    author = zengxq
    time    =14点35分
    '''

# import pandas as pd
# import glob
# import os
# import numpy as np
# import openpyxl
# import torch
#
# if __name__ == '__main__':
#     csvname = r'G:\lung\nii\train_64nor.csv'  # train_hessian_debug.csv Work\Datasets\new_samples\train_only_vessel_new.csv
#     # pathname = r'/data4/zengxq/lung/64x64x32/train_val_nor/'
#     savepath = r'G:\lung\nii\train_32nor.csv'
#     filenamelist=[]
#     numberlist=[]
#     # edgepath=r'/data4/zengxq/64Xsamples/64xsamples/edge/'
#     # hessianpath=r'/data4/zengxq/64Xsamples/64xsamples/hessian/'
#
#     input_df = pd.read_csv(csvname)
#     print("\n sum of val patches is :", input_df.shape[0])
#     for i in range(input_df.shape[0]):
#         labelname =  input_df.iloc[i].at['filename']
#         dataname =  str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'
#         number=input_df.iloc[i].at['filename'].split('__')[-1].split('seg')[0]
#
#         label1=input_df.iloc[i].at['filename'].split('__')[0]+'01'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
#         filenamelist.append(label1)
#         numberlist.append(number)
#
#         label2=input_df.iloc[i].at['filename'].split('__')[0]+'02'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
#         filenamelist.append(label2)
#         numberlist.append(number)
#
#         label3=input_df.iloc[i].at['filename'].split('__')[0]+'03'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
#         filenamelist.append(label3)
#         numberlist.append(number)
#
#         label4=input_df.iloc[i].at['filename'].split('__')[0]+'04'+'__'+input_df.iloc[i].at['filename'].split('__')[-1]
#         filenamelist.append(label4)
#         numberlist.append(number)
#
#         output_excel = { 'filename': [],'number':[]}
#
#         output_excel['filename'] = filenamelist
#         output_excel['number']=numberlist
#         output = pd.DataFrame(output_excel)
#         output.to_csv(csvname, index=False)
#         print('\r[ %d / %d]' % (i, input_df.shape[0]), end='')
