from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from seg_depth_model import *
from img_data import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str, default='./')
parser.add_argument("--images_path", type=str)
parser.add_argument("--seg_path", type=str, default=None)
parser.add_argument("--depth_path", type=str, default=None)
parser.add_argument("--input_shape", type=int, default=(64, 224))

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--finetune", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--n_classes", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--train", type=bool, default=True)
args = parser.parse_args()

images_path_ = args.images_path
seg_path_ = args.seg_path
depth_path_ = args.depth_path
batch_size_ = args.batch_size
n_classes_ = args.n_classes
input_shape_ = args.input_shape
val_ratio_ = args.val_ratio
learning_rate_ = args.learning_rate
finetune_ = args.finetune
train_ = args.train

save_weights_path_ = args.save_weights_path
epochs_ = args.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(left_img_path, right_img_path, seg_path, depth_path, n_classes=10,
          epochs=10, batch_size=5, learning_rate=0.01, val_ratio=0.1, saved_model_path='./', finetune=False, input_shape=input_shape_):

    # model initialized
    sd_model = SegDepthModel(n_channels=3, n_classes=n_classes, n_depth=1)

    # finetune or train
    if finetune:
        # load pretrained model
        sd_model.load_state_dict(torch.load(saved_model_path))
        optimizer = optim.Adam(sd_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(sd_model.parameters(), lr=learning_rate)

    sd_model = sd_model.to(device)

    # adaptive learning rate
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: epoch // 20, lambda epoch: 0.9 ** epoch])

    # construct loss
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    huber_loss = nn.SmoothL1Loss()
    mse_loss = nn.MSELoss()

    # image data initialized
    img_data = RandomBatchImg(left_img_path, right_img_path, seg_path, depth_path, val_ratio=val_ratio,
                              batch_size=batch_size, img_shape=input_shape, pattern_list=('0', '1', '', ''))

    val_left, val_right = img_data.load_stereo_batch(val=True)
    val_seg = img_data.load_seg_batch(val=True)
    val_depth = img_data.load_depth_batch(val=True)
    val_left, val_right, val_seg, val_depth = torch.from_numpy(val_left).to(device), \
                                              torch.from_numpy(val_right).to(device), \
                                              torch.from_numpy(val_seg).to(device), \
                                              torch.from_numpy(val_depth).to(device)

    localtime = time.strftime("%Y-%m-%d-%H-%M")
    # create the training log
    trainlog = open(saved_model_path + '/training_log_' + localtime + '.txt', "a+")
    # calculate the time of training
    start_time = time.time()

    # start training
    for epo in range(1, epochs+1):
        total_loss = 0
        for s in img_data.steps_ind:
            # prepare img data
            batch_left, batch_right = img_data.load_stereo_batch(s)
            batch_seg = img_data.load_seg_batch(s)
            batch_depth = img_data.load_depth_batch(s)
            batch_left, batch_right, batch_seg, batch_depth = torch.from_numpy(batch_left).to(device), \
                                                              torch.from_numpy(batch_right).to(device), \
                                                              torch.from_numpy(batch_seg).to(device), \
                                                              torch.from_numpy(batch_depth).to(device)

            # training
            sd_model.train()
            optimizer.zero_grad()
            seg_out, depth_out = sd_model(batch_left, batch_right)
            loss = ce_loss(seg_out, batch_seg) + dice_loss(seg_out, batch_seg) + mse_loss(depth_out, batch_depth) + huber_loss(depth_out, batch_depth)
            total_loss += loss.item()
            loss.backward()
            scheduler.step()

        # validation
        sd_model.eval()
        seg_out, depth_out = sd_model(val_left, val_right)
        val_loss = ce_loss(seg_out, val_seg) + dice_loss(seg_out, val_seg) + mse_loss(depth_out, val_depth) + huber_loss(depth_out, val_depth)

        # print training message
        slog = 'Train Epoch: {} time: {:.2f}\tLoss: {:.6f}\tval_Loss: {:.6f}'.format(epo, time.time() - start_time, total_loss/img_data.steps_ind[-1], val_loss.item())
        print(slog)
        trainlog.write(slog + '\n')
    trainlog.close()
    # save model
    torch.save(sd_model.state_dict(), saved_model_path)


def test(left_img_path, right_img_path, seg_path, depth_path, batch_size=5, saved_model_path='./'):
    with torch.no_grad():
        sd_model = SegDepthModel()
        sd_model.load_state_dict(torch.load(saved_model_path))
        sd_model = sd_model.to(device)
        sd_model.eval()

        # construct loss
        ce_loss = nn.CrossEntropyLoss()
        dice_loss = DiceLoss()
        huber_loss = nn.SmoothL1Loss()
        mse_loss = nn.MSELoss()

        # image data initialized
        img_data = RandomBatchImg(left_img_path, right_img_path, seg_path, depth_path,
                                  batch_size=batch_size, img_shape=(128, 448), pattern_list=('0', '1', '', ''))

        localtime = time.strftime("%Y-%m-%d-%H-%M")
        vallog = open(saved_model_path + '/validation_log_' + localtime + '.txt', "a+")
        total_loss = 0
        for s in img_data.steps_ind:
            # prepare img data
            batch_left, batch_right = img_data.load_stereo_batch(s)
            batch_seg = img_data.load_seg_batch(s)
            batch_depth = img_data.load_depth_batch(s)
            batch_left, batch_right, batch_seg, batch_depth = torch.from_numpy(batch_left).to(device), \
                                                              torch.from_numpy(batch_right).to(device), \
                                                              torch.from_numpy(batch_seg).to(device), \
                                                              torch.from_numpy(batch_depth).to(device)
            seg_out, depth_out = sd_model(batch_left, batch_right)
            loss = ce_loss(seg_out, batch_seg) + dice_loss(seg_out, batch_seg) + mse_loss(depth_out, batch_depth) + huber_loss(depth_out, batch_depth)
            total_loss += loss.item()

        # print validation message
        slog = 'Validation Loss: {:.6f}'.format(total_loss / img_data.steps_ind[-1])
        print(slog)
        vallog.write(slog + '\n')
        vallog.close()


def main():
    if train_:
        train(images_path_, images_path_, seg_path_, depth_path_, n_classes=n_classes_, epochs=epochs_,
              batch_size=batch_size_,
              learning_rate=learning_rate_, val_ratio=val_ratio_, saved_model_path=save_weights_path_, finetune=finetune_)


if __name__ == '__main__':
    main()
