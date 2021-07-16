#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 00:23:38 2021

@author: anabia
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:04:02 2021

@author: anabia
"""


######### some common Libraries ##########
import os
from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.metrics import recall_score, accuracy_score, precision_score



def performance_report(actual_label, pred_labels):
    recall_=recall_score(actual_label, pred_labels)
    precision_=precision_score(actual_label, pred_labels)
    tn, fp, fn, tp = confusion_matrix(actual_label, pred_labels).ravel()
    specificity_=tn/(tn+fp)
    acc=accuracy_score(actual_label, pred_labels)
    f_score=f1_score(actual_label, pred_labels, pos_label=1,average='binary')
    return f_score, recall_, precision_, acc, specificity_,fn,fp,tn,tp



def classifcation_report(actual_label, pred_labels):
    tn, fp, fn, tp = confusion_matrix(actual_label, pred_labels).ravel()
    return fn,fp,tn,tp


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label') 
    plt.ylabel('True label')  
    plt.tight_layout()
  
  
######### Performance Curves - PR and ROC Curves ################
def ROC_plot(label_list,prediction_prob,col,classifier_n,line_style):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(label_list, prediction_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=col, label=classifier_n+'(AUC: %0.2f)' % (roc_auc), linestyle=line_style)
    plt.plot([0, 1], [0, 1])#, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()


def PR_plot(label_list,prediction_prob,col,classifier_n,line_style):
    plt.figure()
    precision, recall, thresholds =  precision_recall_curve(label_list,prediction_prob)
    pr_auc =average_precision_score(label_list, prediction_prob,average='weighted')
    # step_kwargs = ({'step': 'post'}
    #            if 'step' in signature(plt.fill_between).parameters
    #                else {})
    plt.step(recall, precision, color=col, label=classifier_n+'(AUC: %0.2f)' % (pr_auc), where='post', linestyle=line_style) #alpha=0.2,
    #    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.legend(loc="upper right")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
