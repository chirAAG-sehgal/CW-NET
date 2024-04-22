import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import argparse

from dataloader import Rescale
from dataloader import ToTensor
from dataloader import HumanDataset
from dataloader import RandomFlip

from models import UU2NET
temp_loss = 0
CE_loss = nn.CrossEntropyLoss()
def train(args):
    writer = SummaryWriter()
    epoch_num = 100000
    batch_size_train = 6
    batch_size_val = 1
    train_num = 0
    val_num = 0
    model_name = 'uu2net'
    model_dir = os.path.join(os.getcwd(), 'weights'+ os.sep)
    # dataset_dir = os.path.join(os.getcwd(), "crack_dataset")
    dataset_dir = "sitting_dataset"
    # paths = {"img_path":os.path.join(dataset_dir, "images"), "mask_path": os.path.join(dataset_dir, "masks")}
    paths = {"img_path":os.path.join(dataset_dir, "img"), "mask_path": os.path.join(dataset_dir, "masks")}
    human_dataset = HumanDataset(
        img_path = paths['img_path'],
        mask_path = paths['mask_path'],
        transforms = transforms.Compose([
            Rescale(300),
            RandomFlip(),
            ToTensor(flag = 1)
        ]))
    print(len(human_dataset))
    human_dataloader = DataLoader(human_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    net = UU2NET()
    if torch.cuda.is_available():
        net.cuda()

    print("---define optimizer...")
    net, optimizer = get_model(args)
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0#starting iteration
    running_loss = 0 #starting iteration
    save_frq = 2000 # save the model every 2000 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(human_dataloader):
            ite_num = ite_num + 1

            inputs, labels = data['image'], data['mask']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.long()

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0 = net(inputs_v)
            loss = CE_loss(d0, labels_v)
            temp_loss = loss
            writer.add_scalar('Loss/train', loss.data.item(), ite_num)
            
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()

            # del temporary outputs and loss
            del d0, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f batch_loss: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num, temp_loss))

            if ite_num % save_frq == 0:

                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_%3f.pth" % (ite_num, running_loss / ite_num, temp_loss))
                running_loss = 0.0
                net.train()  # resume train
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description = 'UU2NET')
    parser.add_argument(
        '--model', type = str, default= None, #model name
        help = 'model to be retrained'
    )
    args = parser.parse_args()
    return args

def get_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model:
        model_path = os.path.join(os.getcwd(), f"weights/{args.model}")
        model_dict = torch.load(model_path)
    net = UU2NET().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if args.model:
        net.load_state_dict(model_dict)
        optimizer.load_state_dict(optimizer.state_dict())
        print(f"{args.model} successfully loaded")
    return net, optimizer

if __name__ == "__main__":
    args = parse_args()
    train(args)