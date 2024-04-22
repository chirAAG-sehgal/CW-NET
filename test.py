import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from dataloader import Rescale
from dataloader import ToTensor
from dataloader import HumanDataset
from models import UU2NET # full size version 173.6 MB
import cv2
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(dataset_dir, image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(dataset_dir+os.sep +image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    # cv2_imo = cv2.cvtColor(imo, cv2.COLOR_RGB2BGR)
    

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():
    device = "cuda"
    model_dir = "uu2net_bce_itr_33000_train_0.038824_1.280589.pth"
    dataset_dir = "virtual_dataset"
    prediction_dir = "virtual_dataset"
    os.makedirs(prediction_dir, exist_ok=True)
    test_salobj_dataset = HumanDataset(
        img_path = dataset_dir+"/frames",
        mask_path = dataset_dir+"/masks",
        transforms = transforms.Compose([
            Rescale(300),
            ToTensor(flag = 1)
        ]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    
    # print("...load U2NET---173.6 MB")
    net = UU2NET()
    

    if device == "cuda":
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        start_time = time.time()
        inputs_test= data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        # gray_test = gray_test.type(torch.FloatTensor)
        if device == "cuda":
            inputs_test = Variable(inputs_test.cuda())
            # gray_test = Variable(gray_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
            # gray_test = Variable(gray_test)

        d0 = net(inputs_test)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time}")
        # normalization
        predict_np = (d0.cpu())
        predict_np = predict_np.data.numpy()
        predict_np = predict_np.argmax(axis=1)
        predict_np = predict_np.squeeze()
        print(predict_np.shape)
        mask_visualize = np.zeros((predict_np.shape[0],  predict_np.shape[1], 3))

        colors_bgr = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (128, 0, 0),
            (0, 0, 0)    # Maroon
        ]
        print(len(colors_bgr))
        print(np.unique(predict_np))
        fig, ax = plt.subplots()
        for i in range(8):
            mask_visualize[predict_np == i] = colors_bgr[i]
        mask_visualize = mask_visualize.astype(np.uint8)
        im = ax.imshow(mask_visualize)
        fig.colorbar(im)
        plt.show()
        # save results to test_results folder
        # if not os.path.exists(prediction_dir):
        #     os.makedirs(prediction_dir, exist_ok=True)
        # save_output(dataset_dir, img_name_list[i_test],pred,prediction_dir)

        del d0

if __name__ == "__main__":
    main()
