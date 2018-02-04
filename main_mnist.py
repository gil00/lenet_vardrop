import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.autograd import Variable

import torchnet as tnt
from torchnet.engine import Engine

import torch.utils.data as util_data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from lenet import LeNet
from LSUV import LSUVinit
from utils import cast, data_parallel, print_tensor_dict


import argparse
import os
import json
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Basic Networks')
##########################################################################
########################### Argument Parsing #############################
parser.add_argument('--max_epochs', default=300, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default = 0.01, type=float)
parser.add_argument('--weightDecay', default = 1e-4, type=float)
parser.add_argument('--dropout_rate', default = 0.1, type=float)

parser.add_argument('--dataroot', default='E:/Code/data', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--nthread', default = 1, type=int)
parser.add_argument('--save_path', default = '', type=str)
parser.add_argument('--resume_path', default = '', type=str)

saveimg_epoch = 0
resume_model = ''
last_checkpoint = ''
last_bestpoint = ''
best_pred = -1
opt = parser.parse_args()
##########################################################################
############################# Create Dataset #############################
def create_dataset(opt):
    ds = datasets.MNIST
    convert = transforms.Compose( [ 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
        ])
    trainset = ds( root=opt.dataroot, train=True, download=False, transform=convert )
    train_loader = util_data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nthread, pin_memory=torch.cuda.is_available() ) #pin_memory 유의미한 속도 차이 없음
    
    testset = ds( root=opt.dataroot, train=False, download=False, transform=convert)
    test_loader = util_data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.nthread, pin_memory=torch.cuda.is_available() )
    return train_loader, test_loader

def main() :
    global resume_model
    opt = parser.parse_args()
    ##########################################################################
    ######################  Create Model & Optimizer #########################
    if opt.resume_path != '':
        opt.save_path = opt.resume_path
        resume_model = './{}/model.pt'.format(opt.resume_path)
    else:
        import time
        timetag = time.strftime("MNIST_%Y_%m%d_%H%M%S")
        opt.save_path = str('{}_epochs{}_lr{}_wd{}_dp{:.2f}'.format(timetag, opt.max_epochs, opt.lr, opt.weightDecay, opt.dropout_rate))

    #criterion = nn.CrossEntropyLoss().cuda()
    model = LeNet()
    names = [n for n, p in model.state_dict().items()]
    dict_params = dict(zip(names, model.parameters()))
    model.cuda()
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        optimizer = SGD(model.parameters(), lr, 0.9, weight_decay=opt.weightDecay)
        #optimizer_Adam = Adam( [param for name, param in dict_params.items() if 'conv' in name], lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return optimizer, scheduler

    optimizer, scheduler = create_optimizer(opt, opt.lr)
    opt.start_epoch = 0

    ##########################################################################
    ##########################  Create DataSet ###############################
    train_loader, test_loader = create_dataset( opt )
    ## Optional : INIT LSUV
    sample = torch.rand(1)
    for i,batch in enumerate(train_loader) :
        sample = Variable(cast(batch[0], opt.dtype))
        break
    LSUVinit(model, sample, cuda = True)

    ##########################################################################
    ##########################  Load Model ###################################
    if resume_model != '':
        if os.path.isfile(resume_model):
            print("=> using pre-trained model '{}'".format(resume_model))
            checkpoint = torch.load(resume_model)
            opt.start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    print('parsed options:', vars(opt))
    is_best = False
    val_acc = 0.
    num_classes = 10
    ######################  Print Params ######################################
    print('\nParameters:')
    print_tensor_dict(model.state_dict())
    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nTotal number of parameters:', n_parameters)
    ######################  Criterion #########################################
    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    
    ##########################################################################
    #####################  Logging ###########################################
    def save_checkpoint(state, epoch=0, acc=0., is_best=False):
        global last_checkpoint, last_bestpoint, best_pred
        if last_checkpoint != '':
            os.remove( last_checkpoint )
        last_checkpoint = './{}/model.pt'.format(opt.save_path, epoch, acc)
        torch.save(state, last_checkpoint)
        if is_best:
            best_pred = acc
            if last_bestpoint != '':
                os.remove( last_bestpoint )
            last_bestpoint = './{}/model_best_ep{}_acc{:.3f}.pt'.format(opt.save_path, epoch, acc)
            shutil.copyfile(last_checkpoint, last_bestpoint)
    
    def log(t, state, acc) :
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save_path, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

        global best_pred
        is_best = False
        if best_pred < acc:
            is_best = True
            best_pred = acc
        save_checkpoint({
                'epoch': state['epoch']+1,
                'state_dict': model.state_dict(),
                'best_pred': optimizer.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, state['epoch'], acc, is_best )

    ##########################################################################
    ###########  Train & Validataion Hooking Function ########################
    def on_start(state):
        state['epoch'] = opt.start_epoch
    
    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader)
        epoch = state['epoch'] + 1
        print( '{}/{}'.format(epoch, opt.max_epochs))

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,\
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state, test_acc))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
              (opt.save_path, state['epoch'], opt.max_epochs, test_acc))
        writer.add_scalars('Loss', {'tr ain_loss':train_loss[0], 'test_loss':meter_loss.value()[0]}, state['epoch'] )
        writer.add_scalars('Accuracy', {'train_acc':train_acc[0], 'test_acc':test_acc}, state['epoch'] )
        writer.add_scalars('Learning rate', {'lr':optimizer.param_groups[0]['lr']}, state['epoch'] )
        if state['epoch'] % 2 == 0 or  state['epoch'] == 1:
            for k, v in model.state_dict().items():
                if 'conv' in k:
                    writer.add_histogram(k, v.cpu().numpy().flatten(), state['epoch'], 'fd')
                    if 'conv1.weight' in k :
                        for i in range(v.shape[0]):
                            writer.add_image( 'conv1_ch{}'.format(i), v[i], state['epoch'])
                    if 'conv2.weight' in k :
                        for i in range(v.shape[0]):
                            for j in range(v.shape[1]):
                                writer.add_image( 'conv2_ic{}_oc{}'.format(i,j), v[i][j], state['epoch'])
            
        scheduler.step(meter_loss.value()[0])

    def on_sample(state):
        state['sample'].append(state['train'])

    def h(state):
        sample = state['sample']
        inputs = Variable(cast(sample[0], opt.dtype))
        targets = Variable(cast(sample[1], 'long'))
        input, C1, S2, C2, S4, F1, F2, F3 = model(inputs, sample[2])
        global saveimg_epoch
        if( state['train'] == True ):
            saveimg_epoch = state['epoch']
        if( state['train'] == False and state['t'] == 0):
            if( saveimg_epoch % 2 == 0 or saveimg_epoch == 1 ) :
                img = input.data[0]
                img = img.div_(6).add_(.5)
                writer.add_image('input', img, saveimg_epoch)
                for i in range(C1.data.shape[1]):
                    writer.add_image('C1_ch{}'.format(i),C1.data[0][i], saveimg_epoch)
                for i in range(C2.data.shape[1]):
                    writer.add_image('C2_ch{}'.format(i),C2.data[0][i], saveimg_epoch)
                #if state['epoch'] % 10 == 0 or  state['epoch'] == 1:
                # for k, v in model.state_dict().items():
                #     if 'conv' in k:
                #         writer.add_image(k, v.cpu().numpy().flatten(), state['epoch'], 'fd')
        crss_etrp = F.cross_entropy(F3, targets)
        if sample[2]:
            likelihood, kld = model.loss(input=inputs, target=targets, train=sample[2], average=True)
            loss = likelihood + kld
        else:
            likelihood = model.loss(input=inputs, target=targets, train=sample[2], average=True)
            loss = likelihood
        return loss, F3

    def on_forward(state):
        classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    ##########################################################################
    ###########  Create SummaryWriter & Engine ###############################
    writer = SummaryWriter(opt.save_path)
    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.train(h, train_loader, opt.max_epochs, optimizer)
    writer.close()

        
if __name__ == '__main__':
    main()