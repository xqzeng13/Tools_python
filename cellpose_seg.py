from cellpose import utils, io, models
from PIL import Image
import glob
import os
from tqdm import  tqdm

#调用模型
model = models.Cellpose(gpu=True, model_type='cyto2')
chan = [0,0]
flowThd = 0.4

folder_dir = r"C:\Users\hello\Desktop\iNos-12.26\\"
savepath=folder_dir
filelist=sorted(glob.glob(os.path.join(folder_dir, '*.tif')))
for file in tqdm(filelist,total=len(filelist)):
    filename=file.split('iNos-12.26\\')[-1].replace('.tif','')
    # filename=folder_dir+'RhoA-L-0.tif'
    img = io.imread(file)
    masks, flows, styles, diams = model.eval(img[:,:,0], diameter=30, channels=chan, flow_threshold=flowThd)#0==red;1==green
    masks[masks>0]=255
    bina_mask=Image.fromarray(masks)
    bina_mask.save(savepath+filename+'.png')

# display results
# fig = plt.figure(figsize=(12,5))
# plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
# plt.tight_layout()
# plt.show()
