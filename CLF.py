#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Microsoft VS header
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if os.name == 'nt' or platform == 'win32':
    print("Running on Windows")
    if 'ptvsd' in sys.modules:
        print("Running in Visual Studio")
        try:
            os.chdir(os.path.dirname(__file__))
            print('CurrentDir: ', os.getcwd())
        except:
            pass
#--------------------------------------------------#
    else:
        print("Running outside Visual Studio")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            pass
#########################################################################################################
#########################################################################################################
# Imports
#--------------------------------------------------#
import ast
import copy
import time
import torch
import scipy
import random
import pickle
import scipy.io
import argparse
import subprocess
import numpy as np
import pandas as pd
from numpy import *
from tqdm import tqdm
from pathlib import Path
from random import shuffle
#--------------------------------------------------#
from torch import nn
from torch.utils import data
import torch.optim as optim
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
#--------------------------------------------------#
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler   
from sklearn.preprocessing import LabelEncoder   
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
#--------------------------------------------------#
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import seaborn as sns
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
#--------------------------------------------------#
from tpot import TPOTRegressor
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
#########################################################################################################
#########################################################################################################
# Random Seeds
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#########################################################################################################
#########################################################################################################
# Args
# Data imports
Step_code = "CLF_general"
data_folder = Path("data/")
data_file="kddcup.txt"
#--------------------------------------------------#
# Prediction NN settings
epoch_num=10
batch_size=256
learning_rate=0.00005
NN_type_list=["Reg", "Clf"]
NN_type=NN_type_list[1]
#--------------------------------------------------#
hid_1=4096
hid_2=4096
hid_3=2048
drp_r=0.6
#--------------------------------------------------#
# Results savings
results_folder = Path("results/" + Step_code +"_results/")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
output_file_3 = Step_code + "_all_X_y.p"
output_file_header = Step_code + "_result_"
#########################################################################################################
#########################################################################################################
# Create Temp Folder for Saving Results
print(">>>>> Creating temporary subfolder and clear past empty folders! <<<<<")
now = datetime.now()
#d_t_string = now.strftime("%Y%m%d_%H%M%S")
d_t_string = now.strftime("%m%d-%H%M%S")
#====================================================================================================#
# REMOVE PREVIOUS EMPTY FOLDERS !!!!!
results_folder_contents = os.listdir(results_folder)
for item in results_folder_contents:
    if os.path.isdir(results_folder / item):
        try:
            os.rmdir(results_folder / item)
            print("Remove empty folder " + item + "!")
        except:
            print("Found Non-empty folder " + item + "!")
temp_folder_name = Step_code + "_" + d_t_string + "_" + NN_type
results_sub_folder = results_folder / (temp_folder_name +"/")
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)
print(">>>>> Temporary subfolder created! <<<<<")
#########################################################################################################
#########################################################################################################
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
#--------------------------------------------------#
orig_stdout = sys.stdout
f = open(results_sub_folder / 'print_out.txt', 'w')
sys.stdout = Tee(sys.stdout, f)
print("="*50)
#--------------------------------------------------#
print("Step_code: ", Step_code)
#--------------------------------------------------#
print("epoch_num: ", epoch_num)
print("learning_rate: ", learning_rate)
print("batch_size: ", batch_size)
print("NN_type: ", NN_type)
#--------------------------------------------------#
print("hid_1: ", hid_1)
print("hid_2: ", hid_2)
print("hid_3: ", hid_3)
print("="*50)
#########################################################################################################
#########################################################################################################
# Get Input files
# Get Sequence Embeddings from X03 pickles.
with open( data_folder / data_file, 'rb') as data_txt:
    lines = data_txt.readlines()

X_data=[]
y_data=[]
for one_line in lines:
    X_data.append(one_line.decode("utf-8")[:-2].split(",")[0:6])
    y_data.append("normal" if one_line.decode("utf-8")[:-2].split(",")[-1] == "normal" else "attack")

df_X_data = pd.DataFrame(X_data)
df_X_data = df_X_data.apply(LabelEncoder().fit_transform)
X_data = df_X_data.values.tolist()

#########################################################################################################
#########################################################################################################
# Define a dict since classes read by pytorch shall be {0,1,...C-1}


class2idx={"normal" : 0, "attack" : 1}
idx2class = {v: k for k, v in class2idx.items()}
y_data = [class2idx[one_class] for one_class in y_data]
#print(y_data)

#########################################################################################################
#########################################################################################################
# Data Split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, stratify=y_data, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=42)

print("data set size: ", len(y_data))
print("train set size: ", len(y_train))
print("valid set size: ", len(y_valid))
print("test set size: ", len(y_test))
print("X dimendion: ", len(X_data[0]))

#########################################################################################################
#########################################################################################################
# Normalize Inputs ? Not for count encodings of ECFPs, would cause over-fitting problems
#print("Normalizing Inputs... NOT necessary in this case.")
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_valid = scaler.transform(X_valid)
#X_test = scaler.transform(X_test)
#print("Normalized Inputs... NOT necessary in this case.")

X_train, y_train = np.array(X_train), np.array(y_train)
X_valid, y_valid = np.array(X_valid), np.array(y_valid)
X_test , y_test  = np.array(X_test) , np.array(y_test)


#########################################################################################################
#########################################################################################################
def get_class_distribution(obj):
    count_dict = {
        "normal": 0,
        "attack": 0,
    }
    for i in obj:
        if i == 0: 
            count_dict['normal'] += 1
        elif i == 1: 
            count_dict['attack'] += 1
        else:
            print("Check classes.")
    return count_dict
#====================================================================================================#
'''
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
# Train
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
# Validation
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_valid)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Valid Set')
# Test
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')
#plt.show()
'''
#########################################################################################################
#########################################################################################################
# LoaderClass
class LoaderClass(Dataset):
    def __init__(self, X_data, y_data):
        super(LoaderClass, self).__init__()
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__ (self):
        return self.X_data.shape[0]
#########################################################################################################
#########################################################################################################
train_dataset = LoaderClass(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
valid_dataset = LoaderClass(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long())
test_dataset  = LoaderClass(torch.from_numpy(X_test).float() , torch.from_numpy(y_test ).long())
##====================================================================================================#
## Weighted Sampling
## Because there's a class imbalance, we use stratified split to create our train, validation, and test sets.
## While it helps, it still does not ensure that each mini-batch of our model sees all our classes. 
## We need to over-sample the classes with less number of values. 
## To do that, we use the WeightedRandomSampler.
## First, we obtain a list called target_list which contains all our outputs. 
## This list is then converted to a tensor and shuffled.
##====================================================================================================#
target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

print("target_list: ", target_list)

class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print("class_weights: ", class_weights)

#print("Getting Weighted Sampling.")
#class_weights_all = class_weights[target_list]
#weighted_sampler = WeightedRandomSampler(
#                                         weights=class_weights_all,
#                                         num_samples=len(class_weights_all),
#                                         replacement=True
#                                         )
#print("Done getting Weighted Sampling.")

##====================================================================================================#
#train_loader = DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler)
#valid_loader = DataLoader(dataset = valid_dataset, batch_size=1)
#test_loader  = DataLoader(dataset = test_dataset , batch_size=1)

#########################################################################################################
#########################################################################################################
train_loader = data.DataLoader(LoaderClass(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),batch_size,True)
valid_loader = data.DataLoader(LoaderClass(torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid).long()),batch_size,False)
test_loader  = data.DataLoader(LoaderClass(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()),batch_size,False)


#########################################################################################################
#########################################################################################################

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MulticlassClassification(nn.Module):
    def __init__(self, 
                 num_feature, 
                 num_class,
                 hid_1,
                 hid_2,
                 hid_3,
                 drp_r,
                 ):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, hid_1)
        self.layer_2 = nn.Linear(hid_1, hid_2)
        self.layer_3 = nn.Linear(hid_2, hid_3)
        self.layer_out = nn.Linear(hid_3, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drp_r)
        self.batchnorm1 = nn.BatchNorm1d(hid_1)
        self.batchnorm2 = nn.BatchNorm1d(hid_2)
        self.batchnorm3 = nn.BatchNorm1d(hid_3)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        #x = self.layer_2(x)
        #x = self.batchnorm2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
#====================================================================================================#
EPOCHS = epoch_num
BATCH_SIZE = batch_size
LEARNING_RATE = learning_rate
NUM_FEATURES = X_train.shape[1]
NUM_CLASSES = 2
print("NUM_FEATURES: ", NUM_FEATURES)
#====================================================================================================#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#====================================================================================================#
model = MulticlassClassification(NUM_FEATURES, 
                                 NUM_CLASSES,
                                 hid_1,
                                 hid_2,
                                 hid_3,
                                 drp_r,
                                 )
model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)

#====================================================================================================#
#Before we start our training, define a function to calculate accuracy per epoch.
#This function takes y_pred and y_test as input arguments. 
#We then apply log_softmax to y_pred and extract the class which has a higher probability.
#After that, we compare the the predicted classes and the actual classes to calculate the accuracy.
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

#====================================================================================================#
# Define 2 dictionaries which will store the accuracy/epoch and loss/epoch for both train and validation sets.

accuracy_stats = {
    'train': [],
    "valid": []
}
loss_stats = {
    'train': [],
    "valid": []
}
#====================================================================================================#

print("Begin training.")
for epoch in tqdm(range(1, EPOCHS+1)):

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        #--------------------------------------------------#
        optimizer.zero_grad()
        #--------------------------------------------------#
        y_train_pred = model(X_train_batch)
        #--------------------------------------------------#
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        #--------------------------------------------------#
        train_loss.backward()
        optimizer.step()
        #--------------------------------------------------#
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    # VALIDATION    
    with torch.no_grad():
        #--------------------------------------------------#
        valid_epoch_loss = 0
        valid_epoch_acc = 0
        #--------------------------------------------------#
        model.eval()
        for X_valid_batch, y_valid_batch in valid_loader:
            X_valid_batch, y_valid_batch = X_valid_batch.to(device), y_valid_batch.to(device)
            #--------------------------------------------------#
            y_valid_pred = model(X_valid_batch)
            #--------------------------------------------------#         
            valid_loss = criterion(y_valid_pred, y_valid_batch)
            valid_acc = multi_acc(y_valid_pred, y_valid_batch)
            #--------------------------------------------------#
            valid_epoch_loss += valid_loss.item()
            valid_epoch_acc += valid_acc.item()
    #--------------------------------------------------#
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['valid'].append(valid_epoch_loss/len(valid_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['valid'].append(valid_epoch_acc/len(valid_loader))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    # TESTING 
    y_pred_list = []
    with torch.no_grad():
        #--------------------------------------------------#
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
        #--------------------------------------------------#
        y_pred_list = np.concatenate(y_pred_list)
        #--------------------------------------------------#
        cm=confusion_matrix(y_test, y_pred_list)
        confusion_matrix_df = pd.DataFrame(cm).rename(columns=idx2class, index=idx2class)
        #--------------------------------------------------#
        fig = plt.figure(figsize=(6,4.8))
        #g = sns.heatmap(confusion_matrix_df, cmap="Reds", center=550, annot=True, fmt="d")
        g = sns.heatmap(confusion_matrix_df, cmap="magma", annot=True, fmt="d")
        g.set(xlabel="")
        #plt.show()
        #--------------------------------------------------#
        fig.savefig(results_sub_folder / (output_file_header + "_HeatMapCM_Similarity_Epoch_" + str(epoch)) , dpi=1000 )


    
    print(f'\n Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Valid Loss: {valid_epoch_loss/len(valid_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Valid Acc: {valid_epoch_acc/len(valid_loader):.3f}')


#====================================================================================================#
# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

#====================================================================================================#
#y_pred_list = np.concatenate(y_pred_list)

cm=confusion_matrix(y_test, y_pred_list)
confusion_matrix_df = pd.DataFrame(cm).rename(columns=idx2class, index=idx2class)

plt.figure()
sns.heatmap(confusion_matrix_df, cmap="Reds", center=550, annot=True, fmt="d")
#sns.heatmap(confusion_matrix_df, center=250, annot=True, fmt="d")
plt.show()

print(classification_report(y_test, y_pred_list)) 

y_pred = y_pred_list
y_real = y_test
test_AUC = roc_auc_score(y_real,y_pred)  
fpr,tpr,_ = roc_curve(y_real,y_pred)
roc_auc = auc(fpr,tpr)
#--------------------------------------------------#
lw = 2

fig = plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate',fontsize=18 ,fontweight='bold')
plt.ylabel('True Positive Rate',fontsize=18 ,fontweight='bold')
plt.title('ROC-' + "AUC: " + str(np.round(test_AUC,3)) + ", Epoch: " + str(epoch+1) ,fontsize=18 ,fontweight='bold')
plt.legend(loc="lower right")
#plt.axis('equal')
plt.ylim([-0.05, 1.05])
plt.xlim([-0.05, 1.05])
fig.savefig(    results_sub_folder / (output_file_header + "_ROC-AUC_" + "epoch_" + str(epoch))   ) 
#====================================================================================================#
lr_precision, lr_recall, _ = precision_recall_curve(y_real,y_pred)
no_skill=len(y_real[y_real==1])/len(y_real)
fig = plt.figure(figsize=(6,6))
AUPRC= np.round(metrics.auc(lr_recall, lr_precision),3)

plt.plot( [0,1], [no_skill,no_skill], linestyle='--',label='No Skill')
plt.plot(lr_recall, lr_precision,marker='.', label='Classifier')
plt.xlabel('Recall',fontsize=18 ,fontweight='bold')
plt.ylabel('Precision',fontsize=18 ,fontweight='bold')
plt.title("AU-PRC: " + str(AUPRC) + ", Epoch: " + str(epoch+1) ,fontsize=18 ,fontweight='bold')
plt.legend(loc="lower right")
plt.ylim([no_skill - 0.1, 1.05])
plt.xlim([-0.05, 1.05])
fig.savefig(    results_sub_folder / (output_file_header + "_AU-PRC_" + "epoch_" + str(epoch))    ) 
