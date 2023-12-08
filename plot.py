import glob
import os
from tqdm import  tqdm
import pandas as pd
import random
import matplotlib.pyplot as plt
folder_dir=r'C:\Users\hello\Desktop\206\\'
csvlist=sorted(glob.glob(os.path.join(folder_dir, '*.csv')))
savepath=folder_dir
for csv in tqdm(csvlist,total=len(csvlist)):
    # csvname = r'C:\Users\hello\Desktop\206\206-L+I-0.csv'
    csvname=csv.split('206\\')[-1].replace('.csv','')
    input_df = pd.read_csv(csv)
    x=[]
    y=[]

    for i in range(input_df.shape[0]):
        # dataname = pathname + input_df.iloc[i].at['filename']
        mean = input_df.iloc[i].at['Mean']
        y.append(mean)
        x.append(i)
    # 创建散点图
    plt.scatter(x, y,color='r', marker='o', alpha=0.5)
    '''
    点的大小：使用s参数可以设置点的大小，表示散点的面积。
    透明度：使用alpha参数可以设置点的透明度，值在0到1之间，0
    表示完全透明，1
    表示完全不透明。
    '''
    # 添加标题和轴标签
    plt.title(csvname)
    plt.xlabel('')
    plt.ylabel('Mean')
    # 显示图例
    #plt.legend(['Data points'])
    # 显示散点图
    # plt.show()
    # 保存散点图为PNG文件
    plt.savefig(folder_dir+'plot_'+csvname+'.png')