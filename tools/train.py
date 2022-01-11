import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from models import model as net
from parallel import DataParallelModel, DataParallelCriterion
from SalEval import SalEval
from criteria import SSIM

import os, shutil, time
import numpy as np
from argparse import ArgumentParser

def build_ssim_loss(window_size=11):
    return SSIM(window_size=window_size)


def BCEDiceLoss(inputs, targets):
    #print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    #print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()

    def forward(self, inputs, target, teacher=False):
        #pred1, pred2, pred3, pred4, pred5 = tuple(inputs)
        if isinstance(target, tuple):
           target = target[0]
        target = target[:,0,:,:]
        loss1 = BCEDiceLoss(inputs[:,0,:,:], target)
        loss2 = BCEDiceLoss(inputs[:,1,:,:], target)
        loss3 = BCEDiceLoss(inputs[:,2,:,:], target)
        loss4 = BCEDiceLoss(inputs[:,3,:,:], target)
        loss5 = BCEDiceLoss(inputs[:,4,:,:], target)
       
        return loss1+loss2+loss3+loss4+loss5

@torch.no_grad()
def val(args, val_loader, model, criterion):
    model.eval()

    salEvalVal = SalEval()

    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):
        if args.depth:
            input, target, depth = batched_inputs
        else:
            input, target = batched_inputs
        start_time = time.time()

        if args.onGPU:
            input = input.cuda()
            target = target.cuda()
            if args.depth:
                depth = depth.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        if args.depth:
            depth_var = torch.autograd.Variable(depth).float()
        else:
            depth_var = None

        # run the mdoel
        output, _ = model(input_var, depth_var, test=False)
        loss = criterion(output, target_var)

        #torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)
        salEvalVal.addBatch(output[:,0,:,:], target_var)
        if iter % 5 == 0:
            print('\r[%d/%d] loss: %.3f time: %.3f' % (iter, total_batches, loss.data.item(), time_taken), end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = salEvalVal.getMetric()

    return average_epoch_loss_val, F_beta, MAE

def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()

    salEvalTrain = SalEval()
    epoch_loss = []
    ssim = build_ssim_loss()

    total_batches = len(train_loader)

    for iter, batched_inputs in enumerate(train_loader):
        if args.depth:
            input, target, depth = batched_inputs
        else:
            input, target = batched_inputs
        start_time = time.time()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches, lr_factor=lr_factor)

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()
            if args.depth:
                depth = depth.cuda()
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        if args.depth:
            depth_var = torch.autograd.Variable(depth).float()
        else:
            depth_var = None

        # run the model
        output, depth_output = model(input_var, depth_var, test=False)
        loss = criterion(output, target_var)
        true_depth = depth_var * 0.5 + 0.5
        loss_depth = args.depth_weight * (1 - ssim(depth_output, true_depth))
        loss = loss + loss_depth
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time

        if args.onGPU and torch.cuda.device_count() > 1:
           output = gather(output, 0, dim=0)
        
        # Computing F-measure and MAE on GPU
        with torch.no_grad():
            salEvalTrain.addBatch(output[:,0,:,:] , target_var)
        
        if iter % 5 == 0:
            print('\riteration: [%d/%d] lr: %.7f loss: %.3f depth_loss: %.3f time:%.3f' % (iter+cur_iter, max_batches*args.max_epochs, lr, loss.data.item(), loss_depth.data.item(), time_taken), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = salEvalTrain.getMetric()

    return average_epoch_loss_train, F_beta, MAE, lr

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = iter
        max_iter = max_batches*args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter+1) / 200 + 0.1 * args.lr # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def trainValidateSegmentation(args):
    model = net.MobileSal(pretrained=True)
    args.savedir = args.savedir + '_ep' + str(args.max_epochs) + '/'

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    if args.onGPU and torch.cuda.device_count() > 1:
        model = DataParallelModel(model)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    depthpred_params = sum([np.prod(p.size()) for p in model.idr.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params - depthpred_params))

    mean = [0.406, 0.456, 0.485]  # [103.53,116.28,123.675]
    std = [0.225, 0.224, 0.229] # [57.375,57.12,58.395]
    
    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7./224.*args.inWidth)),
        myTransforms.RandomFlip(),
        #myTransforms.GaussianNoise(),
        myTransforms.ToTensor()
    ])
    

    # for multi-scale training
    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(256, 256),
        myTransforms.RandomCropResize(int(7. / 224. * 256)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(288, 288),
        myTransforms.RandomCropResize(int(7. / 224. * 288)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])
    
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("NJU2K_NLPR_train", file_root=args.file_root, transform=trainDataset_main, use_depth=args.depth)
    train_data_scale1 = myDataLoader.Dataset("NJU2K_NLPR_train", file_root=args.file_root, transform=trainDataset_scale1, use_depth=args.depth)
    train_data_scale2 = myDataLoader.Dataset("NJU2K_NLPR_train", file_root=args.file_root, transform=trainDataset_scale2, use_depth=args.depth)

    trainLoader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    trainLoader_scale1 = torch.utils.data.DataLoader(
        train_data_scale1,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    trainLoader_scale2 = torch.utils.data.DataLoader(
        train_data_scale2,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    val_data = myDataLoader.Dataset("NJU2K_test", file_root=args.file_root, transform=valDataset, use_depth=args.depth)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=10, num_workers=args.num_workers, pin_memory=False)

    # whether use multi-scale training
    if args.ms:
        max_batches = len(trainLoader) + len(trainLoader_scale1) + len(trainLoader_scale2)
    else:
        max_batches = len(trainLoader)
    
    print('For each epoch, we have {} batches'.format(max_batches))
    
    if args.onGPU:
        cudnn.benchmark = True

    start_epoch = 0
    cur_iter = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write("\n%s\t%s\t%s" % ('Epoch', 'F_beta (val)', 'MAE (val)'))
    logger.flush()
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)


    criteria = DeepSupervisionLoss()
    if args.onGPU and torch.cuda.device_count() > 1:
        criteria = DataParallelCriterion(criteria)


    for epoch in range(start_epoch, args.max_epochs):

        if args.ms:
            train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader)
            torch.cuda.empty_cache()

            train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader)
            torch.cuda.empty_cache()
        
        lossTr, F_beta_tr, MAE_tr, lr = \
            train(args, trainLoader, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()
        
        # evaluate on validation set
        if epoch == 0:
            continue
        
        lossVal, F_beta_val, MAE_val = val(args, valLoader, model, criteria)
        torch.cuda.empty_cache()
        logger.write("\n%d\t\t%.4f\t\t%.4f" % (epoch, F_beta_val, MAE_val))
        logger.flush()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': F_beta_tr,
            'F_val': F_beta_val,
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        # save the model also
        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        if epoch % 1 == 0 and epoch > args.max_epochs * 0.6:
            torch.save(model.state_dict(), model_file_name)

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F_beta(tr) = %.4f\t F_beta(val) = %.4f" \
                % (epoch, lossTr, lossVal, F_beta_tr, F_beta_val))
        torch.cuda.empty_cache()
    logger.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="./data/", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=320, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=320, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=60, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--depth', type=int, default=1, help='use RGB-D data, default True')
    parser.add_argument('--depth_weight', type=float, default=0.3, help='idr loss weight, default 0.3')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')


    args = parser.parse_args()
    print('Called with args:')
    print(args)

    trainValidateSegmentation(args)
