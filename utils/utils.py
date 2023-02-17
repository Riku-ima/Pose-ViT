import os
import torch
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import numpy as np
import sklearn
import pandas as pd


### save_weights

def save_weights(model,args,epoch,optim):
    state={
        'epoch':epoch,
        'state_dict':model.state_dict(),
        'optimizer':optim.state_dict()}
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    model_name='{}_{}epoch'.format(args.base_model,epoch)
    torch.save(state,'{}/{}.pt'.format(args.model_path,model_name))
    return model_name


"""
def load_weights(model,args):
    #epoch=args.start_epoch
    epoch=1
    pretrained=torch.load('{}/{}_{}.pt'.format(args.model_path,args.base_model,epoch))['state_dict']
    
    model_dict=model.state_dict()
    pretrained={k: v for k,v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    
    return model
"""
def write_history(
                    args,
                    epoch,
                    train_loss,train_acc,train_f1,train_precision,train_recall,train_confusion_matrix,
                    val_loss,val_acc,val_f1,val_precision,val_recall,val_confusion_matrix,best_acc):
    #.txt fileに出力  trainの情報+confusion_matrix ,val ....
    """
    hitory_path: history path for save
    """
    history_path=args.history_path+args.base_model+'.txt'
    #初回は内容初期化
    if epoch == 1:
        with open(history_path,'w') as file:
            pass
        
    with open(history_path,'a') as file:
        #ループ初回は書き込み
        if epoch ==1:
            file.write(
                '=========================  {}  =========================\n'.format(args.base_model)
                +str(args)
                      )
            
        
        file.write('checkpoint_name:{} {}epoch lr:{}\n'.format(args.base_model,epoch,args.lr)
        +'train_loss: {} || train_acc: {} || train_f1: {} || train_precision: {} || train_recall: {}\n'.format(
        round(train_loss,5),round(train_acc,5),round(train_f1,5),round(train_precision,5),round(train_recall,5))
        +train_confusion_matrix+'\n'
                   
        + 'val_loss: {} || val_acc: {} || val_f1: {} || val_precision: {} || val_recall: {}\n'.format(
            round(val_loss,5),round(val_acc,5),round(val_f1,5),round(val_precision,5),round(val_recall,5))
        +val_confusion_matrix+'\n'    
        +'BEST VAL Acc :   {:.4f}\n'.format(best_acc)
                  )


def Get_scores(pred_classes,ground_truths,score_type='macro'):
    #get Acc , Precision , recall , F
    """
    input:  
          pred_classes
          ground_truths
          score_type : in [None,'macro','micro',]
          
    """
    
    #acc=sklern.metrics.accuracy_score(ground_truths,pred_classes)
    #precision=sklearn.metrics.precision_score(ground_truths,pred_classes,average=score_type)
    #recall=sklearn.metrics.recall_score(ground_truths,pred_classes,average=score_type)
    #f1=sklearn.metrics.f1_score(ground_truths,pred_classes,average=score_type)
    
    acc=metrics.accuracy_score(ground_truths,pred_classes)
    precision=metrics.precision_score(ground_truths,pred_classes,average=score_type)
    recall=metrics.recall_score(ground_truths,pred_classes,average=score_type)
    f1=metrics.f1_score(ground_truths,pred_classes,average=score_type)
    
    

    
    return acc,precision,recall,f1