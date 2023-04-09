import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import joblib
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from tensorboardX import SummaryWriter
import io

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--size_m', type=int, default=56)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--decay_rates',nargs='+')
parser.add_argument('--decay_epoch',nargs='+')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.5)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--comment', type=str, default='./experiment1')
parser.add_argument('--classweight', type=str, default='false')
parser.add_argument('--debug', action='store_true')
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--input', '-i', default='data/stft_mfcc/b n.p', type=str,
#         help='path to directory with input data archives')
parser.add_argument('--input', '-i', default='data/stft_mfcc/mfcc_stft_train.p', type=str,
        help='path to directory with input data archives')
parser.add_argument('--test', default='data/bi_test_pack/wavelet_stft_test.p', type=str,
        help='path to directory with test data archives')
args = parser.parse_args()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.droupout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
#        print("input:"+str(x.size()))
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.droupout(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.droupout(out)
#        print("output:"+str(out.size()))
        return out + shortcut


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # shape = torch.prod(torch.tensor(x.shape[1:])).item()
        shape = torch.prod(torch.tensor(np.array(x.shape[1:]))).item()
        return x.view(-1, shape)


class BilinearCNN(nn.Module):
    
    def __init__(self,dim):
        super(BilinearCNN, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 64, 3, 1)
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.ResNet_0_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_0 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_1_1 = ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        self.ResNet_0 = ResBlock(64, 64)
        self.ResNet_1 = ResBlock(64, 64)
        self.ResNet_2 = ResBlock(64, 64)
        self.ResNet_3 = ResBlock(64, 64)
        self.ResNet_4 = ResBlock(64, 64)
        self.ResNet_5 = ResBlock(64, 64)
        self.ResNet_6 = ResBlock(64, 64)
        self.ResNet_7 = ResBlock(64, 64)
        self.ResNet_8 = ResBlock(64, 64)
        self.ResNet_9 = ResBlock(64, 64)
        self.ResNet_10 = ResBlock(64, 64)
        self.ResNet_11 = ResBlock(64, 64)
        self.ResNet_12 = ResBlock(64, 64)
        self.ResNet_13 = ResBlock(64, 64)
        self.ResNet_14 = ResBlock(64, 64)
        self.ResNet_15 = ResBlock(64, 64)
        self.ResNet_16 = ResBlock(64, 64)
        self.ResNet_17 = ResBlock(64, 64)
        self.norm0 = norm(dim)
        self.norm1 = norm(dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 4)
        self.dropout = nn.Dropout(args.dropout)
        self.flat = Flatten()
        
        
    def forward(self,stft,mfcc):
        
        out_s = self.conv0(stft)
        out_s = self.ResNet_0_0(out_s)
        out_s = self.ResNet_0_1(out_s)
        out_s = self.ResNet_0(out_s)
        out_s = self.ResNet_2(out_s)
        out_s = self.ResNet_4(out_s)
        out_s = self.ResNet_6(out_s)
        out_s = self.ResNet_8(out_s)
        out_s = self.ResNet_10(out_s)
        out_s = self.ResNet_12(out_s)
#        out_s = self.ResNet_14(out_s)
#        out_s = self.ResNet_16(out_s)
        out_s = self.norm0(out_s)
        out_s = self.relu0(out_s)
        out_s = self.pool0(out_s)
        
        out_m = self.conv1(mfcc)
        out_m = self.ResNet_1_0(out_m)
        out_m = self.ResNet_1_1(out_m)
        out_m = self.ResNet_1(out_m)
        out_m = self.ResNet_3(out_m)
        out_m = self.ResNet_5(out_m)
        out_m = self.ResNet_7(out_m)
        out_m = self.ResNet_9(out_m)
        out_m = self.ResNet_11(out_m)
        out_m = self.ResNet_13(out_m)
#        out_m = self.ResNet_15(out_m)
#        out_m = self.ResNet_17(out_m)
        out_m = self.norm1(out_m)
        out_m = self.relu1(out_m)
        out_m = self.pool1(out_m)

        out = torch.matmul(out_s,out_m)

#        out = torch.bmm(out_s, torch.transpose(out_m, 1, 2))
        out = self.flat(out)
        out = self.linear(out)
        out = self.dropout(out)
       
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class myDataset(data.Dataset):
    def __init__(self, stft, mfcc, targets):
        self.stft = stft
        self.mfcc = mfcc
        self.targets = targets

    def __getitem__(self, index):
        sample_stft = self.stft[index]
        sample_mfcc = self.mfcc[index]
        target = self.targets[index]
        min_s = np.min(sample_stft)
        max_s = np.max(sample_stft)
        sample_stft = (sample_stft-min_s)/(max_s-min_s) 
        min_m = np.min(sample_mfcc)
        max_m = np.max(sample_mfcc)
        sample_mfcc = (sample_mfcc-min_m)/(max_m-min_m) 
        
        output_stft = torch.FloatTensor(np.array([sample_stft]))
        crop_s = transforms.Resize([args.size,args.size])
        img_s = transforms.ToPILImage()(output_stft)
        croped_img=crop_s(img_s)
        output_stft = transforms.ToTensor()(croped_img)
        
        output_mfcc = torch.FloatTensor(np.array([sample_mfcc]))
        crop_m = transforms.Resize([args.size_m,args.size_m])
        img_m = transforms.ToPILImage()(output_mfcc)
        croped_img_m=crop_m(img_m)
        output_mfcc = transforms.ToTensor()(croped_img_m)
#        print(output.size())
        

        return output_stft,output_mfcc,target
    def __len__(self):
        return len(self.stft)

        
def get_mnist_loaders(data_aug=False, batch_size=1, test_batch_size=1000, perc=1.0):
    stft_test, mfcc_test, labels_test = joblib.load(open(args.test, mode='rb'))

    test_loader = DataLoader(
        myDataset(stft_test, mfcc_test, labels_test),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    return test_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    target_flag = 0 #indicating for assuming normal label
    prediction_flag = 0 # indicating for assuming normal prediction
    for stft,mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        if y.item() == 1:
            target_flag = 1
        y = one_hot(np.array(y.numpy()), 2)
        target_class = np.argmax(y, axis=1)
        # predicted_y = model(stft,mfcc).cpu().detach().numpy()
        # print(predicted_y)
        predicted_class = np.argmax(model(stft,mfcc).cpu().detach().numpy(), axis=1)
        if predicted_class[0] == 1:
            prediction_flag = 1

        total_correct += np.sum(predicted_class == target_class)
    # return total_correct / len(dataset_loader.dataset)
    return target_flag, prediction_flag
 
def Loss(model, dataset_loader):
    total_loss = 0
    for stft,mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = y.to(device)
        logits = model(stft,mfcc)
        entroy = nn.CrossEntropyLoss().to(device)
        loss = entroy(logits, y).cpu().numpy()

        total_loss += loss
    return total_loss  / (len(dataset_loader.dataset)/args.batch_size)

def confusion_matrix(model, dataset_loader):
    targets = []
    outputs = []

    for stft,mfcc, y in dataset_loader:
        stft = stft.to(device)
        mfcc = mfcc.to(device)
        y = one_hot(np.array(y.numpy()), 2)
        # y = one_hot(np.array(y.numpy()), 4)

        target_class = np.argmax(y, axis=1)
        targets = np.append(targets,target_class)
        predicted_class = np.argmax(model(stft,mfcc).cpu().detach().numpy(), axis=1)
        outputs = np.append(outputs,predicted_class)

        
    Confusion_matrix=sk_confusion_matrix(targets.tolist(), outputs.tolist())
    print('Confusion_matrix:')
    print(Confusion_matrix)
    target_names = ['normal', 'abnormal']
    print('classification_report:')
    print(classification_report(targets.tolist(), outputs.tolist(), target_names=target_names))
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BilinearCNN(64).to(device)
    dict = torch.load('./model/model_split.pth')
    state_dict =dict['state_dict']
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict, strict = True)
    model.eval()
    target_list = []
    prediction_list = []
    normal_list = []
    normal_prediction_list = []
    total_num = 0
    correct_num = 0
    # test_loader = get_mnist_loaders()
    test_path = './data/bi_test_pack/hospital_pack/'
    files = os.listdir(test_path)
    files.sort()

    for file in tqdm(files):
        args.test = test_path+file
        test_loader = get_mnist_loaders()
        # confusion_matrix(model, test_loader)
        # val_acc = accuracy(model, test_loader)
        target, prediction = accuracy(model,test_loader)
        target_list.append(target)
        prediction_list.append(prediction)
    print('Target Results are as follwoing:')
    print(target_list)
    print()
    print('Prediction are as follwoing:')
    print(prediction_list)
    for i in range(len(target_list)):
        total_num += 1
        if target_list[i] == 0:
            normal_list.append(target_list[i])
            normal_prediction_list.append(prediction_list[i])
        if target_list[i] == prediction_list[i]:
            correct_num+=1
    print('Accuracy is {}'.format(correct_num/total_num))
    print(total_num)

    total_num = 0
    correct_num = 0
    for i in range(len(normal_list)):
        total_num += 1
        if normal_prediction_list[i] == 0:
            correct_num +=1
    print('The target list of normal is {}'.format(normal_list))
    print()
    print('The prediction list of normal is {}'.format(normal_prediction_list))
    print(f'Accuracy of predicting Normal is {correct_num/total_num}')
    print(total_num)




    # for i, data in tqdm(enumerate(test_loader)):
    #     val_acc = accuracy(model, test_loader)
    #     print(val_acc)    
        # confusion_matrix(model, test_loader)

    # for i, data in tqdm(enumerate(test_loader)):
    #     val_acc = accuracy(model, test_loader)
    #     print(val_acc)
    