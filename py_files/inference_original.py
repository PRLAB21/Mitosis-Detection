#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 00:05:11 2021

@author: anabia
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:51:45 2021

@author: anabia
"""


####### torch Libraries ##########
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F

######### some common Libraries ##########
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as  pd


from py_files.performance_measure import *
from py_files.RHINet import *
from py_files.ASTMNet import *
from py_files.DSTMNet import *
from py_files.ATTENNet import *
from py_files.ResidualNet import *


def dir_creat(path):
    if not os.path.exists(path):
        os.mkdir(path)


def data_loader(img_dir, batch=1, size=120):
    class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            path, target = self.samples[index]
            name = self.samples[index][0] # it returns the path of the image
            tuple_with_path = (original_tuple + (path,))
            # return tuple_with_path
            return tuple_with_path #sample, target,original_tuple
    transform_data = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolderWithPaths(root=img_dir, transform=transform_data)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=1)
    return loader





#############  in this below section define the folder and name of the dataset - data path and reading  #############

                # classifier_nam should be  ["RHINet","ASTMNet", "DSTMNet", "ATTENNet", "ResidualNet"]
                # img_size = {"RHINet": 120, "ASTMNet": 120, "DSTMNet": 120, "ATTENNet": 120, "ResidualNet": 224}
                # performance report should be [performance_report, classifcation_report]
                # performance report should be [plot_confusion_matrix, ROC_plot, PR_plot]

TAG = '[inference.py]'
# base_dir = "/home/anabia/Documents/github_codes/"
base_dir = os.getcwd()
# data_path = os.path.join(base_dir, 'dataset')
data_path = os.path.join(base_dir, 'example_img')
model_path = os.path.join(base_dir, 'trained_models')  
output_folder_path = os.path.join(base_dir, 'predicted_output')
dir_creat(output_folder_path)
output_report = 'classifcation_report'
performance_plot = ['plot_confusion_matrix', 'PR_plot']
classifier_nam = "ASTMNet" 
batch_size = 1
device = 'cpu'
output_csv = os.path.join(output_folder_path, classifier_nam + "_output.csv")
performance_report_csv = os.path.join(output_folder_path, classifier_nam + output_report + "output.csv")



if 'RHINet' == classifier_nam:
    image_size = 120
    model = RHINet().to(device)

elif 'ASTMNet' == classifier_nam:
    image_size = 120
    model = ASTMNet().to(device)

elif 'DSTMNet' == classifier_nam:
    image_size = 120
    model = DSTMNet().to(device)

elif 'ATTENNet' == classifier_nam:
    image_size = 120
    model = ATTENNet().to(device)

elif 'ResidualNet' == classifier_nam:
    image_size = 224
    model = ResidualNet().to(device)
    
else:
    raise ModuleNotFoundError


loader = data_loader(data_path, batch=batch_size, size=image_size)
checkpoint = torch.load(os.path.join(model_path, classifier_nam + ".ckpt"), map_location=device)
model.load_state_dict(checkpoint)
print(TAG, '[model]\n', model)


prediction_prob = []
classifier_output = []
score_list = []
list_perf = []
total_actual_labels = []
total_predicted_labels = []
total = 0
correct = 0
class_label = {0: 'non-mitosis', 1: 'mitosis'}

with torch.no_grad():
    model.eval()
    for v_images, v_labels, path in loader:
        v_images = v_images.to(device)
        v_labels = v_labels.to(device)
        print(TAG, '[v_images]\n', v_images)
        print(TAG, '[v_labels]\n', v_labels)
        name = os.path.basename(path[0])
        print(TAG, '[name]\n', name)
        class_name = path[0].split("/")[-2]
        print(TAG, '[class_name]\n', class_name)

        pred_outputs = model(v_images)
        print(TAG, '[pred_outputs]\n', pred_outputs)
        score = pred_outputs.data.cpu().numpy().tolist()[0][1]
        print(TAG, '[score]\n', score)
        score_list.append(score) 
        prob = F.softmax(pred_outputs, dim=1)
        print(TAG, '[prob]\n', prob)
        score, predicted_label = torch.max(pred_outputs, 1)
        print(TAG, '[score]\n', score)
        print(TAG, '[predicted_label]\n', predicted_label)
        pred_outputs = pred_outputs.data.cpu().numpy().tolist()    
        print(TAG, '[pred_outputs]\n', pred_outputs)

        predicted_label = predicted_label.data.cpu().numpy().tolist() 
        print(TAG, '[predicted_label]\n', predicted_label)
        labels = v_labels.data.cpu().numpy().tolist()
        prob = prob.data.cpu().numpy().tolist()[0]

        columns = ['image name', 'non-mitosis score', 'mitosis score', 'non-mitosis-probability', 'mitosis-probability', 'Actual class', 'Predicted class', 'Actual label', 'Predicted label']
        list_perf.append([name] + [pred_outputs[0][0]] + [pred_outputs[0][1]] + [prob[0]] + [prob[1]] + [class_label[labels[0]]] + [class_label[predicted_label[0]]] + [labels[0]] + [predicted_label[0]])
        props_df = pd.DataFrame(data=list_perf)
        props_df.to_csv(output_csv, index=False, header=columns)

        total_actual_labels = total_actual_labels + labels 
        total_predicted_labels = total_predicted_labels + predicted_label

        break

    if output_report == "performance_report":
        f_score, recall_, precision_, acc, specificity_,PFN,PFP,PTN,PTP= performance_report(total_actual_labels, total_predicted_labels)
        pred_correct = PTN+PTP
        total_img = PTN+PTP+PFP+PFN
        
        columns = ['classifier name', 'FN', 'FP', 'TN', 'TP', 'f_score', 'recall', 'precision', 'accuracy', 'specificity']
        classifier_output = [ [classifier_nam] +[PFN]+[PFP]+[PTN]+[PTP] + [f_score] + [recall_] + [precision_] + [acc]+ [specificity_]]
        props_df = pd.DataFrame(data=classifier_output)
        props_df.to_csv(performance_report_csv, index=False, header=columns)
        
    
    if output_report == "classifcation_report":
        PFN,PFP,PTN,PTP = classifcation_report(total_actual_labels, total_predicted_labels)
        pred_correct = PTN + PTP
        total_img = PTN + PTP + PFP + PFN
        
        columns = ['classifier name',  'FN', 'FP', 'TN', 'TP']
        classifier_output = [[classifier_nam] + [PFN] + [PFP] + [PTN] + [PTP]]
        props_df = pd.DataFrame(data=classifier_output)
        props_df.to_csv(performance_report_csv, index=False, header=columns)


for plot in performance_plot:
    if plot == 'plot_confusion_matrix':
        cnf_matrix = confusion_matrix(total_actual_labels, total_predicted_labels)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix, without normalization')
        
    if plot == 'ROC_plot':
        ROC_plot(total_actual_labels,score_list,col='b',classifier_n=classifier_nam,line_style='-')

    if plot == 'PR_plot':
        PR_plot(total_actual_labels,score_list,col='b',classifier_n=classifier_nam, line_style='-')


print('Test Accuracy of the model on the {} test images: {:.2f}%'.format(total_img, 100*(pred_correct/total_img)))
print ("F1 Score: ", f1_score(total_actual_labels, total_predicted_labels, pos_label=1,average='binary'))
