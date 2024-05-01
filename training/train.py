import numpy as np
import pandas as pd
import cv2
import os
import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
# import seaborn as sns
import random
import sys
# from efficientnet_pytorch import EfficientNet
from sklearn.metrics import mean_absolute_error
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import gmtime, strftime
from sklearn.model_selection import train_test_split
import albumentations
from albumentations.pytorch import ToTensorV2
import copy
import timm

# !pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git tqdm albumentations timm

# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc

from torch.utils.data import WeightedRandomSampler

import utils

import json
import sys


test_transform = albumentations.Compose([
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def get_data_df(CFG):
    train_df = pd.read_csv(CFG['path_train_csv'])
    valid_df = pd.read_csv(CFG['path_val_csv'])
    test_df = pd.read_csv(CFG['path_test_csv'])

    crop_around_disc = True if 'crop_around_disc' in CFG.keys() and CFG['crop_around_disc'] else False

    superiorly_only=True if 'superiorly_only' in CFG.keys() and CFG['superiorly_only'] else False
    inferiorly_only=True if 'inferiorly_only' in CFG.keys() and CFG['inferiorly_only'] else False
    
    if crop_around_disc or superiorly_only or inferiorly_only:
        train_df = train_df[~train_df['disc_x1'].isna()]
        valid_df = valid_df[~valid_df['disc_x1'].isna()]
        test_df = test_df[~test_df['disc_x1'].isna()]
    
    if CFG['DEBUG_MODE']:
        DEBUG_MODE_ratio = CFG['DEBUG_MODE_ratio']
        train_df =  train_df.sample(int(len(train_df)*DEBUG_MODE_ratio))
        valid_df =  valid_df.sample(int(len(valid_df)*DEBUG_MODE_ratio))
        test_df  =  test_df.sample(int(len(test_df)* DEBUG_MODE_ratio))
    return train_df, valid_df, test_df


def load_model(CFG, num_classes=11):
    model = timm.create_model(CFG['model_name'], pretrained=True,num_classes=num_classes)
    
    if 'efficient' in CFG['model_name']:    
        model.classifier = nn.Sequential(model.classifier, nn.Sigmoid())
    elif 'convnext' in CFG['model_name']:   
        model.head.fc = nn.Sequential(model.head.fc, nn.Sigmoid())
    elif 'vit' in CFG['model_name']:   
        model.head = nn.Sequential(model.head, nn.Sigmoid())
    elif 'eva' in CFG['model_name']:   
        model.head = nn.Sequential(model.head, nn.Sigmoid())
    else:
        raise NotImplementedError    
    
    model.cuda();
    if CFG['DataParallel']:
        model = nn.DataParallel(model);
        
    return model  

def plot_auc_roc(fpr_list, tpr_list, roc_auc_list, figsize=None, lw_list=[1], color_list=['darkorange'], name_list=['name'], title='ROC'):
    if figsize: 
        f, ax = plt.subplots(figsize=fig_size)
    else:
        f, ax = plt.subplots()
    for i in range(len(fpr_list)):       
        ax.plot(
            fpr_list[i],
            tpr_list[i],
            color=color_list[i],
            lw=lw_list[i],
            label=f"{name_list[i]} ROC curve (area = %0.5f)" % roc_auc_list[i],
            # linestyle=
        )
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return f, ax

def train_model(train_loader, model, optimizer, criterion, age_auxiliary_labels=False):
    model.train() 
        
    avg_loss = 0.
    optimizer.zero_grad()
    for idx, (imgs,  labels, _, auxiliary_labels) in enumerate(tqdm(train_loader)):
        imgs_train, labels_train, auxiliary_labels_train = imgs.cuda(), labels.float().cuda(), auxiliary_labels.float().cuda()

        output_train = model(imgs_train)
        output_train_sigmoid = output_train
        
        # loss = criterion(output_train_sigmoid,labels_train)
        if not age_auxiliary_labels:
            loss = criterion(output_train_sigmoid,labels_train)
        else:
            loss = criterion(output_train_sigmoid,torch.hstack([labels_train,auxiliary_labels_train]))     
        loss.backward()
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()
        optimizer.step() 
        optimizer.zero_grad() 
        avg_loss += loss.item() / len(train_loader)
        
    return avg_loss


def test_model(test_loader, model, optimizer, criterion, age_auxiliary_labels=False):
    avg_test_loss = 0.
    model.eval()
    preds = []
    gt = []
    with torch.no_grad():
        for idx, (imgs, labels_test, _, auxiliary_labels)  in enumerate(tqdm(test_loader)):
            imgs_vaild, labels_test, auxiliary_labels_test = imgs.cuda(), labels_test.float().cuda(), auxiliary_labels.float().cuda()
            output_test = model(imgs_vaild)
            output_test_sigmoid = output_test # torch.sigmoid(output_test)

            preds.append(output_test_sigmoid.to('cpu').numpy())
            gt.append(labels_test.to('cpu').numpy())
            
            if not age_auxiliary_labels:
                test_loss = criterion(output_test_sigmoid,labels_test)
            else:
                test_loss = criterion(output_test_sigmoid,torch.hstack([labels_test,auxiliary_labels_test]))
                
            avg_test_loss += test_loss.item() / len(test_loader)
    
    predictions = np.concatenate(preds)     
    gt = np.concatenate(gt)
    return avg_test_loss, predictions, gt

    

def run(cfg_path):  
    with open(cfg_path) as f:
        CFG = json.load(f)
    print(CFG)


    
    utils.seed_everything(CFG['seed'])    
    train_df, valid_df, test_df = get_data_df(CFG)
    

    
    if CFG['weighted_sampler']:
        if CFG['weighted_sampler_label'] == 'Final Label':
            sampler = utils.balanced_sampler(train_df[CFG['weighted_sampler_label']])
        if CFG['weighted_sampler_label'] == 'Smooth Final Label':
            sampler = utils.smooth_balanced_sampler((train_df[CFG['weighted_sampler_label']] > 0).astype(float))

    crop_around_disc = True if 'crop_around_disc' in CFG.keys() and CFG['crop_around_disc'] else False
    if crop_around_disc:
        crop_around_disc_size = CFG['crop_around_disc_crop_ratio']
    else:
        crop_around_disc_size = 8

    superiorly_only=True if 'superiorly_only' in CFG.keys() and CFG['superiorly_only'] else False
    inferiorly_only=True if 'inferiorly_only' in CFG.keys() and CFG['inferiorly_only'] else False

    rotate_augmentation_limit = CFG['rotate_augmentation_limit'] if 'rotate_augmentation_limit' in CFG.keys() else 60
    train_transform = albumentations.Compose([
        albumentations.CLAHE(), 
        albumentations.HorizontalFlip(),
        albumentations.Rotate(limit=rotate_augmentation_limit, border_mode=0),
        albumentations.Sharpen(alpha=(0.1,0.5)),
        albumentations.RandomContrast(limit=(0.1, 0.25)),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


    trainset     = utils.JustRAIGSDataset(
        train_df,                
        target_label=CFG['train_label_header'], 
        justification_labels=CFG['train_justification_headers'], 
        transform =train_transform,  
        IMG_SIZE=(CFG['IMG_SIZE'],CFG['IMG_SIZE']), 
        crop_around_disc=crop_around_disc, 
        crop_around_disc_size=crop_around_disc_size,
        superiorly_only=superiorly_only,
        inferiorly_only=inferiorly_only,
    ) 
    
    train_loader = torch.utils.data.DataLoader(
        trainset,                        
        batch_size=CFG['BATCH_SIZE'], 
        shuffle=False if CFG['weighted_sampler'] else True, 
        num_workers=CFG['num_workers'],        
        pin_memory=False,         
        sampler=sampler if CFG['weighted_sampler'] else None)
    
    valset = utils.JustRAIGSDataset(
        valid_df, 
        target_label=CFG['val_label_header'], 
        justification_labels=CFG['val_justification_headers'], 
        transform=test_transform, 
        IMG_SIZE=(CFG['IMG_SIZE'], CFG['IMG_SIZE']),         
        crop_around_disc=crop_around_disc, 
        crop_around_disc_size=crop_around_disc_size,
        superiorly_only=superiorly_only,
        inferiorly_only=inferiorly_only,    
    )
    
    val_loader = torch.utils.data.DataLoader(
        valset, 
        batch_size=CFG['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=CFG['num_workers'], 
        pin_memory=False)

    print('len(trainset)',len(trainset))
    print('len(valset)',len(valset))
    print('trainset.get_number_of_classes()', trainset.get_number_of_classes())
    print('valset.get_number_of_classes()', valset.get_number_of_classes())
    
    if 'age_auxiliary_labels' in CFG.keys() and CFG['age_auxiliary_labels']:
        num_classes = trainset.get_number_of_classes() + 10
    else: 
        num_classes = trainset.get_number_of_classes()
    
    model = load_model(CFG, num_classes)    

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)

    
    # pos_weights_train = torch.Tensor([0.03224120761562959, 0.019926446663971684, 0.02219417686323615, 0.0018930617315599026, 0.0020803959654121847, 0.0032536998511185824, 0.0033522968163039943, 0.005097463100085779, 0.0008084951145203751, 0.00842018082683415, 0.011220334638099839])
    # pos_weights_test = torch.Tensor([0.142, 0.08769748336190991, 0.0976778995782579, 0.008331477884951363, 0.009155947050649675, 0.014319727614760153, 0.014753658754601372, 0.022434239929790906, 0.0035582353466979775, 0.03705771934243992, 0.04938136371393047])
    
    # if 'pos_weight' in CFG and CFG['pos_weight'] == 'trainset_ratios':
    #     pos_weight = pos_weights_train.cuda()
    # elif 'pos_weight' in CFG and CFG['pos_weight'] == 'testset_ratios':
    #     pos_weight = pos_weights_test.cuda() 
    # else:
    #     pos_weight=None

    criterion = nn.BCELoss()

        
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.15)
    warmup_epo = CFG['warmup_epo']
    cosine_epo = CFG['cosine_epo'] 
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = utils.GradualWarmupSchedulerV2(optimizer, multiplier=2, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    early_stopper = utils.EarlyStopper(patience=CFG['early_stopping_patience'])


    trial_dir_path = utils.create_expriment_dir(
        CFG['model_name'],              
        debug=CFG['DEBUG_MODE'],           
        crop_disc=True if 'crop_around_disc' in CFG.keys() and CFG['crop_around_disc'] else False
    )

    best_score = -1 # sensitivity_at_desired_specificity
    n_epochs = warmup_epo + cosine_epo
    validation_report_list = list()
    score_dict_history_list = list()
    
    os.makedirs(trial_dir_path, exist_ok=True)
    with open(os.path.join(trial_dir_path, cfg_path.split(os.sep)[-1]), 'w') as f:
        json.dump(CFG, f)
    
    for epoch in range(n_epochs):
        scheduler_warmup.step(epoch)
        lr =  scheduler_warmup.get_lr()[0]
        print('lr:',lr) 
        start_time   = time.time()
        age_auxiliary_labels = True if 'age_auxiliary_labels' in CFG.keys() and CFG['age_auxiliary_labels'] else False
        
        avg_train_loss = train_model(train_loader, model, optimizer, criterion, age_auxiliary_labels=age_auxiliary_labels)
        avg_val_loss, preds, valid_labels = test_model(val_loader, model, optimizer, criterion, age_auxiliary_labels=age_auxiliary_labels)
            
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t train_loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_train_loss, avg_val_loss, elapsed_time))
        
        # score_dict = get_score(preds.reshape(-1),valid_labels.reshape(-1))
        score_dict = utils.get_score(y_pred_prob = preds[:,0:11], valid_labels = valid_labels, justification_label_names=['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC'])
        score_dict_history_list.append(score_dict)
        
        valid_sensitivity_at_desired_specificity = score_dict['sensitivity_at_desired_specificity_final']
        
        print("Score : ", score_dict['sensitivity_at_desired_specificity_final'])
        score_dict_summary =         {
                "epoch": epoch, 
                "elapsed_time":elapsed_time, 
                "sensitivity_at_desired_specificity":score_dict['sensitivity_at_desired_specificity_final'], 
                'threshold_at_desired_specificity':score_dict['threshold_at_desired_specificity_final'], 
                "roc_auc":score_dict['roc_auc_final'],
                "train_loss":avg_train_loss, 
                "val_loss":avg_val_loss, 
                "lr":lr
            }
        
        suffix_list = [f'_{suffix}'for suffix in ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']]
        print(score_dict.keys())
        for suffix in suffix_list:
            score_dict_summary[f'hamming_loss{suffix}'] = score_dict[f'hamming_loss{suffix}']
            score_dict_summary[f'roc_auc{suffix}'] = score_dict[f'roc_auc{suffix}'] 
        score_dict_summary['hamming_loss'] = score_dict['hamming_loss']
        score_dict_summary['hamming_loss_threshold'] = score_dict['hamming_loss_threshold']
        validation_report_list.append(score_dict_summary)    
        pd.DataFrame(validation_report_list).to_csv(os.path.join(trial_dir_path, 'train_report_df.csv'), index=False)
    
        # save according to score (higher = better)
        print('best_score', best_score)
        print('sensitivity_at_desired_specificity', valid_sensitivity_at_desired_specificity)
        if valid_sensitivity_at_desired_specificity > best_score:
            print('improved!, saving..')
            best_score = valid_sensitivity_at_desired_specificity
            checkpoint_save_dest = os.path.join(trial_dir_path, f'epoch_{epoch}_weight_best_{valid_sensitivity_at_desired_specificity}.pt')
        else:
            checkpoint_save_dest = os.path.join(trial_dir_path, f'epoch_{epoch}_{valid_sensitivity_at_desired_specificity}.pt')

        if CFG['DataParallel']:
            torch.save(model.module.state_dict(), checkpoint_save_dest)
        else:
            torch.save(model.state_dict(), checkpoint_save_dest)

        np.save(checkpoint_save_dest.replace('.pt','_test_proba_without_tta.npy'), preds)

        
        name_list = ['final', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
        
        f, ax = plot_auc_roc(
            fpr_list=[score_dict[f'fpr_{suffix}'] for suffix in name_list], 
            tpr_list=[score_dict[f'tpr_{suffix}'] for suffix in name_list], 
            roc_auc_list=[score_dict[f'roc_auc_{suffix}'] for suffix in name_list],
            color_list = ['k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            lw_list = [1 for _ in range(len(name_list))],
            name_list = name_list,                 
        )
        # plt.show()
        plt.savefig( os.path.join(trial_dir_path,f'figure_epoch_{epoch}.png'));
        # plt.show()
        plt.clf();
        f.clear();
        plt.close(f);
    
        
        print(validation_report_list[-1])
        # break
        print(early_stopper)
        early_stopping = early_stopper.early_stop(valid_sensitivity_at_desired_specificity)
        print(early_stopper)
        print(f'early_stopping = {early_stopping}')
        if early_stopping:  
            print(f'early_stopping -> break train loop')        
            break
    
    train_report_df = pd.DataFrame(validation_report_list)
    train_report_df.to_csv(os.path.join(trial_dir_path, 'train_report_df.csv'), index=False)

if __name__ == "__main__":
    cfg_path = sys.argv[1] # 'train_configs_9/DEBUG/DEBUG__efficientnet_b5__IMGSIZE_896__fold_0__lr_5e06__final_header_Eval__justification_header_Eval.json' 
    print(cfg_path)
    run(cfg_path)