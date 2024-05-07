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
from matplotlib import pyplot as plt
from time import gmtime, strftime
from sklearn.model_selection import train_test_split
import albumentations
from albumentations.pytorch import ToTensorV2
import copy
import timm

# !pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git tqdm albumentations timm
import torchvision.transforms

# import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, hamming_loss

from torch.utils.data import WeightedRandomSampler
import json



def balanced_sampler(y_train):
    class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
    print('class_sample_count', class_sample_count)
    weight = 1. / class_sample_count
    print('weight', weight)
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))  
    return sampler
    
def smooth_balanced_sampler(y_train):   
    negative_class_weight = 1 / len(y_train[y_train == 0])
    smooth_class_weight = 1 / len(y_train[y_train > 0])
    
    samples_weight = np.array([smooth_class_weight if t>0 else negative_class_weight for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)) 
    return sampler

def get_sensitivity_at_desired_specificity(y_pred_prob, y_true, desired_specificity=0.95, suffix='_final'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    idx = np.argmax(fpr >= (1 - desired_specificity))
    threshold_at_desired_specificity = thresholds[idx]    
    sensitivity_at_desired_specificity = tpr[idx]
    return {
        f'sensitivity_at_desired_specificity{suffix}':sensitivity_at_desired_specificity, 
        f'roc_auc{suffix}':roc_auc, 
        f'fpr{suffix}':fpr, 
        f'tpr{suffix}':tpr, 
        f'threshold_at_desired_specificity{suffix}':threshold_at_desired_specificity}


def get_hamming_loss(y_pred_prob, y_true, headers = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC'], hamming_th_step=0.0001, hamming_on_positives_only=True):
    assert y_true.shape[1] == 11
    assert y_pred_prob.shape[1] == 11   

    if hamming_on_positives_only:
        positive_rows = np.argwhere(y_true[:,0] > 0)
    else:
        positive_rows = np.argwhere(gt[:,0] == gt[:,0])
        
    best_th = np.ones_like(np.array(range(10))) * 0.6
    best_hamming_loss = hamming_loss(y_true[positive_rows][:,0,1:], y_pred_prob[positive_rows][:,0,1:] > best_th)
    for target_column_idx in range(len(headers)):
        # print('iterating over best th, col', str(target_column_idx))
        for th in np.arange(0, 1, hamming_th_step):
            current_th_array = best_th.copy()
            current_th_array[target_column_idx] = th
            
            loss = hamming_loss(y_true[positive_rows][:,0,1:], y_pred_prob[positive_rows][:,0,1:] > current_th_array)
            # print(loss, th)
            
            if loss < best_hamming_loss:
                # print('new best', loss)
                best_hamming_loss = loss
                best_th[target_column_idx] = th
    
    output_ = {
        'hamming_loss':best_hamming_loss,
        'hamming_loss_threshold':best_th,        
    }

    for ii in range(len(headers)):
        # print(headers[ii])
        loss_per_header = hamming_loss(y_true=y_true[positive_rows][:,0,ii+1:ii+2],  y_pred=y_pred_prob[positive_rows][:,0,ii+1:ii+2] > current_th_array[ii:ii+1])
        output_[f'hamming_loss_{headers[ii]}'] = loss_per_header
        # print(headers[ii],loss_per_header )
    return output_


def get_score(y_pred_prob, valid_labels, justification_label_names=['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC'], hamming_th_step=0.0001, hamming_on_positives_only=True):
    if not justification_label_names:
        justification_label_names = []        
    final_label_preds = y_pred_prob[:, 0:1]
    final_label_y_true = valid_labels[:, 0:1]

    justification_labels_preds = y_pred_prob[:, 1:]
    justification_labels_y_true = valid_labels[:, 1:]
    
    score_dict_final_label = get_sensitivity_at_desired_specificity(y_pred_prob=final_label_preds, y_true=final_label_y_true, desired_specificity=0.95)
    score_dict_justification_labels = get_hamming_loss(y_pred_prob=y_pred_prob, y_true=valid_labels, hamming_th_step=hamming_th_step, hamming_on_positives_only=hamming_on_positives_only)
    
    score_dict = dict()
    score_dict.update(score_dict_final_label)
    for idx, justification_label in enumerate(justification_label_names):
        score_dict.update(get_sensitivity_at_desired_specificity(
            y_pred_prob=y_pred_prob[:, idx+1], 
            y_true=valid_labels[:, idx+1], 
            desired_specificity=0.95,
            suffix=f'_{justification_label}',
        ))
    score_dict.update(score_dict_justification_labels)
    
    return score_dict


def get_date_str():
    return strftime("%y%m%d_%H_%M", gmtime())

def create_expriment_dir(model_name, debug=False, crop_disc=False):

    main_dir_name = 'log'
    os.makedirs(main_dir_name, exist_ok=True) 
    date_str = get_date_str()
    
    if debug:
        date_str = f'{date_str}_debug'
        
    if crop_disc:
        date_str = f'{date_str}_disc_crop'
        
    trial_dir_path = os.path.join(main_dir_name, model_name, date_str)
    os.makedirs(trial_dir_path, exist_ok=True) 
    return trial_dir_path

def plot_auc_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 1
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.5f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.max_validation_score = -float('inf')

    def __str__(self):
        return f'patience = {self.patience}, counter = {self.counter}, max_validation_score = {self.max_validation_score}'

    def early_stop(self, validation_score):
        if validation_score > self.max_validation_score:
            self.max_validation_score = validation_score
            self.counter = 0
        elif validation_score <= self.max_validation_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        
def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X    
    
class JustRAIGSDataset(Dataset):
    def __init__(self, df, target_label, justification_labels=None, 
                 transform=None, 
                 IMG_SIZE=(512, 512), 
                 crop_around_disc=False, crop_around_disc_size=8,
                 superiorly_only=False,
                 inferiorly_only=False,

                ):
        self.df = df
        self.transform = transform
        self.IMG_SIZE = IMG_SIZE
        self.target_label=target_label
        self.justification_labels = justification_labels
        self.crop_around_disc = crop_around_disc
        self.crop_around_disc_size = crop_around_disc_size

        self.superiorly_only = superiorly_only
        self.inferiorly_only = inferiorly_only
        if self.superiorly_only and self.inferiorly_only:
            raise NotImplementedError

        cols_to_zero_on_superiorly_crop = ['Eval ANRI', 'Eval RNFLDI', 'Eval BCLVI', 'Eval NVT', 'Eval DH', 'Eval LD', 'Eval LC', 'Smooth Only RG ANRI', 'Smooth Only RG RNFLDI', 'Smooth Only RG BCLVI', 'Smooth Only RG NVT', 'Smooth Only RG DH', 'Smooth Only RG LD', 'Smooth Only RG LC', 'Smooth Include NRG ANRI', 'Smooth Include NRG RNFLDI', 'Smooth Include NRG BCLVI', 'Smooth Include NRG NVT', 'Smooth Include NRG DH', 'Smooth Include NRG LD', 'Smooth Include NRG LC']
        cols_to_zero_on_inferiorly_crop = ['Eval ANRS', 'Eval RNFLDS', 'Eval BCLVS', 'Eval NVT', 'Eval DH', 'Eval LD', 'Eval LC', 'Smooth Only RG ANRS', 'Smooth Only RG RNFLDS', 'Smooth Only RG BCLVS', 'Smooth Only RG NVT', 'Smooth Only RG DH', 'Smooth Only RG LD', 'Smooth Only RG LC', 'Smooth Include NRG ANRS', 'Smooth Include NRG RNFLDS', 'Smooth Include NRG BCLVS', 'Smooth Include NRG NVT', 'Smooth Include NRG DH', 'Smooth Include NRG LD', 'Smooth Include NRG LC']
        if self.superiorly_only:
            for col in cols_to_zero_on_superiorly_crop:
                self.df[col] = 0
        if self.inferiorly_only:
            for col in cols_to_zero_on_inferiorly_crop:
                self.df[col] = 0
    
    def __len__(self):
        return len(self.df)

    def get_number_of_classes(self):
        return 1 + len(self.justification_labels) 
    
    def __getitem__(self, idx):
        
        final_label = self.df[self.target_label].values[idx]
        final_label = np.expand_dims(final_label, -1)

        labels_justification_array = []
        if self.justification_labels:
            labels_justification_array = self.df[self.justification_labels].values[idx]

        # print('self.justification_labels', self.justification_labels)
        # print('labels_justification_array', labels_justification_array)

        # age_bin_labels = np.array(self.df[[f'Ages_{a}_{a+10}' for a in list(range(0,100,10))]].values[idx])
        # auxiliary_labels = age_bin_labels
        
        p_path = self.df.path.values[idx]
        # print(p_path)
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        if self.crop_around_disc:
            disc_x1 = int(self.df['disc_x1'].values[idx])
            disc_x2 = int(self.df['disc_x2'].values[idx])
            disc_y1 = int(self.df['disc_y1'].values[idx])
            disc_y2 = int(self.df['disc_y2'].values[idx])
            disc_x =  int(self.df['disc_x'].values[idx])
            disc_y =  int(self.df['disc_y'].values[idx])
            x_max, y_max = max(disc_x1, disc_x2), max(disc_y1, disc_y2)
            x_min, y_min = min(disc_x1, disc_x2), min(disc_y1, disc_y2)
            crop_size = image.shape[0]/self.crop_around_disc_size

            image = np.array(Image.fromarray(image).crop((x_min-crop_size, y_min-crop_size, 
                                                x_max+crop_size, y_max+crop_size)))
        else:
            image = crop_image_from_gray(image)

        ####
        if self.superiorly_only or self.inferiorly_only:
            disc_y = int(self.df['disc_y'].values[idx])  
        if self.superiorly_only :
            image = shift_image(image, 0, image.shape[0] - disc_y)
        if self.inferiorly_only :
            image = shift_image(image, 0, -disc_y)        
        ####

        
        image = cv2.resize(image, self.IMG_SIZE)
        
        if self.transform and type(self.transform) == albumentations.core.composition.Compose:
            image = self.transform(image=image)['image']
        elif self.transform and type(self.transform) == torchvision.transforms.transforms.Compose:
            image = self.transform(image)
        else:
            image = self.transform(image=image)['image']

        if self.justification_labels:
            label = np.hstack([final_label, labels_justification_array])
        else:
            label = final_label
                
        return image, label, final_label, labels_justification_array

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
