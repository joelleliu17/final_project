"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from logging import debug
import os
import time
import math
from config import get_args
from utils.utils import get_logger
from utils.cli_utils import *
from model.models import resnet50, KANC_MLP_Big
from methods import tent, deyo
import time
import wandb

import warnings


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()
args.method = 'tent' ## or 'deyo' or 'noadapt'


logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False) 

def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f')
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')
        
    model.eval()

    with torch.no_grad():
        end = time.time()
        correct_count = [0,0,0,0]
        total_count = [1e-6,1e-6,1e-6,1e-6]
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            if biased:
                if args.dset=='Waterbirds':
                    place = dl[2]['place'].cuda()
                else:
                    place = dl[2].cuda()
                group = 2*target + place #0: landbird+land, 1: landbird+sea, 2: seabird+land, 3: seabird+sea
                
            # compute output
            if args.method=='deyo':
                output = adapt_model(images, i, target, flag=False, group=group)
            else:
                output = model(images)
            # measure accuracy and record loss
            if biased:
                TFtensor = (output.argmax(dim=1) == target)
                for group_idx in range(4):
                    correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                    total_count[group_idx] += len(TFtensor[group==group_idx])
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
                

            # '''
            if (i+1) % args.wandb_interval == 0:
                if biased:
                    LL = correct_count[0]/total_count[0]*100
                    LS = correct_count[1]/total_count[1]*100
                    SL = correct_count[2]/total_count[2]*100
                    SS = correct_count[3]/total_count[3]*100
                    LL_AM.update(LL, images.size(0))
                    LS_AM.update(LS, images.size(0))
                    SL_AM.update(SL, images.size(0))
                    SS_AM.update(SS, images.size(0))
                    if args.wandb_log:
                        wandb.log({f'{args.corruption}/LL': LL,
                                   f'{args.corruption}/LS': LS,
                                   f'{args.corruption}/SL': SL,
                                   f'{args.corruption}/SS': SS,
                                  })
                if args.wandb_log:
                    wandb.log({f'{args.corruption}/top1': top1.avg,
                               f'{args.corruption}/top5': top5.avg})
                
                progress.display(i)
            # '''
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            if (i+1) % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break
            '''
            
    if biased:
        acc1s, acc5s = [], []
        LLs, LSs, SLs, SSs = [], [], [], []

        logger.info(f"- Detailed result under {args.corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
        if args.wandb_log:
            wandb.log({'final_avg/LL': LL,
                       'final_avg/LS': LS,
                       'final_avg/SL': SL,
                       'final_avg/SS': SS,
                       'final_avg/AVG': (LL+LS+SL+SS)/4,
                       'final_avg/WORST': min(LL,LS,SL,SS)
                      })
            
        avg = (LL+LS+SL+SS)/4
        logger.info(f"Result under {args.corruption}. The adaptation accuracy of {args.method} is  average: {avg:.5f}")

        LLs.append(LL)
        LSs.append(LS)
        SLs.append(SL)
        SSs.append(SS)
        acc1s.append(avg)
        acc5s.append(min(LL,LS,SL,SS))

        logger.info(f"The LL accuracy are {LLs}")
        logger.info(f"The LS accuracy are {LSs}")
        logger.info(f"The SL accuracy are {SLs}")
        logger.info(f"The SS accuracy are {SSs}")
        logger.info(f"The average accuracy are {acc1s}")
        logger.info(f"The worst accuracy are {acc5s}")
    else:
        logger.info(f"Result under {args.corruption}. The adaptation accuracy of {args.method} is top1: {top1.avg:.5f} and top5: {top5.avg:.5f}")

        acc1s.append(top1.avg.item())
        acc5s.append(top5.avg.item())

        logger.info(f"acc1s are {acc1s}")
        logger.info(f"acc5s are {acc5s}")
    return top1.avg, top5.avg



#################### Cifar-100-C dataset loading ######################

'''
也可以替換成其他dataset
'''


import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from PIL import Image


corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',        ## Noise
          'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', ## Blur
          'snow', 'frost', 'fog', 'brightness',           ## Weather
          'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', ## Digital

          ## Extra 4 corruptions
          'speckle_noise', 'gaussian_blur', 'spatter', 'saturate' ]


# CIFAR-100-C 資料路徑
# cifar100c_root = 'C:/Users/User/Desktop/dataset/TTA/CIFAR-100-C'
cifar100c_root = args.data_root


# CIFAR-100-C 測試資料集載入（你需要事先把 CIFAR-100-C 下載好放在 cifar100c_root）
class CIFAR100C_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        # CIFAR-100-C corruption data 存成 numpy file, e.g. gaussian_noise.npy
        corrupted_file = os.path.join(root, f"{corruption}.npy")
        self.data = np.load(corrupted_file)
        self.data = self.data[(severity-1)*10000 : severity*10000]  # 取對應 severity 的 10000 張圖片
        self.transform = transform
        # 讀取標籤
        self.targets = np.load(os.path.join(root, 'labels.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # CIFAR-100-C 裡的圖片是 uint8 numpy array，轉成 PIL Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target

# 定義 transform（跟訓練時一樣）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])

# 建立 CIFAR-100-C 測試資料載入器
args.severity = 5                   ## severity = int(config('severity'))  # 1~5
args.corruption = 'gaussian_noise'  ## corruption is one of corruptions

test_dataset = CIFAR100C_Dataset(cifar100c_root, corruption=args.corruption, severity=args.severity, transform=transform_test)  
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)  ## batch_size default 128



#################### loaded pretrained model ######################

## choose model
args.model = 'resnet50'

if args.model == 'resnet50':
    # pretrained_model = torch.load('resnet50-cifar100.pth', weights_only=False)

    pretrained_model = resnet50()
    pretrained_model.load_state_dict(torch.load('resnet50-cifar100.pth'))

elif args.model == 'kan':
    pretrained_model = KANC_MLP_Big()
    pretrained_model.load_state_dict(torch.load('KAN-cifar100.pth'))

# elif args.model == 'm':
#     pretrained_model = 
#     pretrained_model.load_state_dict(torch.load('-cifar100.pth'))


#################### some args change #####################

args.num_class = len(np.unique(test_dataset.targets))  ## 計算有幾類

bs = args.test_batch_size
args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
args.lr *= args.lr_mul


## deyo
args.deyo_margin *= math.log(args.num_class) # for thresholding
args.deyo_margin_e0 *= math.log(args.num_class) # for reweighting tuning





#################### start to adapt ######################

if args.method == 'noadapt':
    print(f"method: {args.method}, corruption: {args.corruption}")

    model = pretrained_model.to(device)
    model.eval()
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    logits_test, targets_test= [], []
    with torch.no_grad():
        for i, dl in enumerate(test_loader):
            images, target = dl[0], dl[1]
            images, target = images.to(device), target.to(device)

            outputs = model(images)
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            
            logits_test.append(outputs.cpu().detach().numpy())
            targets_test.append(target.cpu().detach().numpy())  

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    print(f"acc1s are {top1.avg.item()}")
    print(f"acc5s are {top5.avg.item()}")


    logits_test = np.concatenate(logits_test, axis=0)
    targets_test = np.concatenate(targets_test, axis=0)

    logits = torch.tensor(logits_test, dtype=torch.float32)
    targets = torch.tensor(targets_test, dtype=torch.long)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    ac = correct / targets.size(0)

    print(f'ACC = {ac:.4f}')



elif args.method == 'tent':
    print(f"method: {args.method}, corruption: {args.corruption}")

    st_time = time.time()

    model = pretrained_model.to(device)
    net = tent.configure_model(model.to(device))
    params, param_names = tent.collect_params(net)

    if not params:
        print("There is no BatchNorm, LayerNorm or GroupNorm on pre-trained model.")
        
    # print(param_names)

    optimizer = torch.optim.SGD(params, float(args.lr), momentum=0.9)
    tented_model = tent.Tent(net, optimizer)

    # acc1, acc5 = validate(test_loader, tented_model, device, mode='eval')
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    logits_test, targets_test= [], []
    for i, dl in enumerate(test_loader):
        images, target = dl[0], dl[1]
        images, target = images.to(device), target.to(device)

        outputs = tented_model(images)
        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            
        logits_test.append(outputs.cpu().detach().numpy())
        targets_test.append(target.cpu().detach().numpy())  

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()


    en_time = time.time()

    print(f'time: {en_time - st_time}')
    print(f"acc1s are {top1.avg.item()}")
    print(f"acc5s are {top5.avg.item()}")

    print(f"acc1s are {top1.avg.item()}")
    print(f"acc5s are {top5.avg.item()}")

    logits_test = np.concatenate(logits_test, axis=0)
    targets_test = np.concatenate(targets_test, axis=0)

    logits = torch.tensor(logits_test, dtype=torch.float32)
    targets = torch.tensor(targets_test, dtype=torch.long)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    ac = correct / targets.size(0)

    print(f'ACC = {ac:.4f}')



elif args.method == 'deyo':

    print(f"method: {args.method}, aug_type: {args.aug_type}, corruption: {args.corruption}")
    model = pretrained_model.to(device)
    
    biased = False
    wandb_log = False       

    st_time = time.time()

    net = deyo.configure_model(model.to(device))
    params, param_names = deyo.collect_params(net)
    # print(param_names)

    if not params:
        print("There is no BatchNorm, LayerNorm or GroupNorm on pre-trained model.")

    optimizer = torch.optim.SGD(params, float(args.lr), momentum=0.9)
    adapt_model = deyo.DeYO(net, args, optimizer=optimizer, deyo_margin= args.deyo_margin, margin_e0= args.deyo_margin_e0)


    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f')
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, top1, top5],
            prefix='Test: ')
    end = time.time()
    count_backward = 1e-6
    final_count_backward =1e-6
    count_corr_pl_1 = 0
    count_corr_pl_2 = 0
    total_count_backward = 1e-6
    total_final_count_backward =1e-6
    total_count_corr_pl_1 = 0
    total_count_corr_pl_2 = 0
    correct_count = [0,0,0,0]
    total_count = [1e-6,1e-6,1e-6,1e-6]

    logits_test, targets_test= [], []
    for i, dl in enumerate(test_loader):
        images, target = dl[0], dl[1]
        images, target = images.to(device), target.to(device)

        if biased:
            place = dl[2].cuda()
            group = 2*target + place
        else:
            group=None

        output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)  ## adapt

        if biased:
            TFtensor = (output.argmax(dim=1)==target)
            
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                total_count[group_idx] += len(TFtensor[group==group_idx])
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        logits_test.append(output.cpu().detach().numpy())
        targets_test.append(target.cpu().detach().numpy())  
             
        count_backward += backward
        final_count_backward += final_backward
        total_count_backward += backward
        total_final_count_backward += final_backward
        
        count_corr_pl_1 += corr_pl_1
        count_corr_pl_2 += corr_pl_2
        total_count_corr_pl_1 += corr_pl_1
        total_count_corr_pl_2 += corr_pl_2

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        if (i+1) % int(args.wandb_interval) == 0:
            if biased:
                LL = correct_count[0]/total_count[0]*100
                LS = correct_count[1]/total_count[1]*100
                SL = correct_count[2]/total_count[2]*100
                SS = correct_count[3]/total_count[3]*100
                LL_AM.update(LL, images.size(0))
                LS_AM.update(LS, images.size(0))
                SL_AM.update(SL, images.size(0))
                SS_AM.update(SS, images.size(0))
                if wandb_log:
                    wandb.log({f"{args.corruption}/LL": LL,
                                f"{args.corruption}/LS": LS,
                                f"{args.corruption}/SL": SL,
                                f"{args.corruption}/SS": SS,
                                })

            if wandb_log:
                wandb.log({f'{args.corruption}/top1': top1.avg,
                            f'{args.corruption}/top5': top5.avg,
                            f'acc_pl_1': count_corr_pl_1/count_backward,
                            f'acc_pl_2': count_corr_pl_2/final_count_backward,
                            f'count_backward': count_backward,
                            f'final_count_backward': final_count_backward})
            
            count_backward = 1e-6
            final_count_backward =1e-6
            count_corr_pl_1 = 0
            count_corr_pl_2 = 0

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % int(args.wandb_interval) == 0:
            progress.display(i)

    acc1 = top1.avg
    acc5 = top5.avg
    
    if biased:
        print(f"- Detailed result under {args.corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
        if wandb_log:
            wandb.log({'final_avg/LL': LL,
                        'final_avg/LS': LS,
                        'final_avg/SL': SL,
                        'final_avg/SS': SS,
                        'final_avg/AVG': (LL+LS+SL+SS)/4,
                        'final_avg/WORST': min(LL,LS,SL,SS),
                        })
        
    if wandb_log:
        wandb.log({f'{args.corruption}/top1': acc1,
                    f'{args.corruption}/top5': acc5,
                    f'total_acc_pl_1': total_count_corr_pl_1/total_count_backward,
                    f'total_acc_pl_2': total_count_corr_pl_2/total_final_count_backward,
                    f'total_count_backward': total_count_backward,
                    f'total_final_count_backward': total_final_count_backward})

    if biased:
        avg = (LL+LS+SL+SS)/4
        print(f"Result under {args.corruption}. The adaptation accuracy of DeYO is  average: {avg:.5f}")

        # LLs.append(LL)
        # LSs.append(LS)
        # SLs.append(SL)
        # SSs.append(SS)
        # acc1s.append(avg)
        # acc5s.append(min(LL,LS,SL,SS))

        # print(f"The LL accuracy are {LLs}")
        # print(f"The LS accuracy are {LSs}")
        # print(f"The SL accuracy are {SLs}")
        # print(f"The SS accuracy are {SSs}")
        # print(f"The average accuracy are {acc1s}")
        # print(f"The worst accuracy are {acc5s}")
    else:
        en_time = time.time()

        print(f'time: {en_time - st_time}')
        print(f"acc1s are {top1.avg.item()}")
        print(f"acc5s are {top5.avg.item()}")

    logits_test = np.concatenate(logits_test, axis=0)
    targets_test = np.concatenate(targets_test, axis=0)

    logits = torch.tensor(logits_test, dtype=torch.float32)
    targets = torch.tensor(targets_test, dtype=torch.long)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    ac = correct / targets.size(0)


    # num_classes = int(config('num_class'))
    # ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
    # brier = brier_score(logits_test, targets_test, num_classes)
    # AUROC = calculate_auroc_multiclass(logits_test, targets_test, num_classes)

    # print( '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f, MCE = %.4f, Brier = %.5f, AUROC = %.4f' %(ac, ECE, MCE, brier, AUROC) )  

    print(f'ACC = {ac:.4f}')