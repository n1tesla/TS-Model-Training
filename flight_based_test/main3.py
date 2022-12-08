import pandas as pd
import numpy as np
import glob
import os
import tensorflow as tf
import pickle
import joblib
from functions_list import *
import sys
import warnings
import matplotlib
import random
from pathlib import Path
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import classification_report
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def define_paths(mode):
    cwd = os.getcwd()
    model_path = cwd + "\models"
    if mode:
        data_path = os.path.dirname(cwd) + "\\data\\test_data"
    else:
        data_path = os.path.dirname(cwd) + "\\data\\test_fusion_dataset"
    data_path_list = []

    for file in glob.glob(data_path + "/*"):
        data_path_list.append(file)

    model_path_list = []
    for file in os.listdir(model_path):
        print(file)
    for file in glob.glob(model_path + "/*"):
        if file == model_path + '\\ModelResults':
            continue
        model_path_list.append(file)
    model_result_save_path = glob.glob(model_path + "/ModelR*")

    return model_path_list,data_path_list,model_result_save_path

def parse_data(data_path,mode):
    test_data=pd.read_csv(data_path+"\\extractedData_All_Train.csv") #TODO : extractedData_All_Test olmalı ismi. hem ecodyne hem acar modu düzgün çalışması için

    if mode:
        echo_neg_index=test_data[test_data["probUAV"]<0.5].index
        test_data.loc[echo_neg_index,"probUAV"]=0
        echo_pos_index=test_data[test_data["probUAV"]>0.5].index
        test_data.loc[echo_pos_index,"probUAV"]=1
        echodyne_data=test_data[["pos_x","pos_y","label","probUAV"]]
        echodyne_data.rename(columns={"probUAV":"pred_label"},inplace=True)
        return test_data,echodyne_data
    else:
        try:
            acar_pos=pd.read_csv(data_path+"\\acar_pos.csv")
        except FileNotFoundError:
            acar_pos=create_fake_acar_df()
        acar_pos["label"]=1
        try:
            acar_neg=pd.read_csv(data_path+"\\acar_neg.csv")
        except FileNotFoundError:
            acar_neg=create_fake_acar_df()


        acar_neg["label"]=0
        acar_data=pd.concat([acar_pos,acar_neg])
        acar_data.rename(columns={'P_UAV_labeled':'pred_label','X_labeled':'pos_x','Y_labeled':'pos_y'},inplace=True)

        return test_data,acar_data

def create_fake_acar_df():
    pos_x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    pos_y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    df=pd.DataFrame(columns=['pos_x', 'pos_y'])
    df['P_UAV_labeled']=[0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1]
    df['pos_x']=pos_x
    df['pos_y']=pos_y
    return df


def prepare_data(test_data,scaler,feature_list,window_size,stride_size,model_shape):
    test_data.loc[:,feature_list]=scaler.transform(test_data.loc[:,feature_list].values)
    WindowedX,WindowedY,PosXY=overlap_TrackID(test_data,'id',window_size,stride_size,feature_list,model_shape)

    return WindowedX,WindowedY,PosXY

def ihtar_model(model_path):
    config=pd.read_csv(model_path+"\\config.csv")
    scaler=joblib.load(model_path+"\\scaler.bin")
    model=tf.keras.models.load_model(model_path)
    window_size=config["Window_Size"][0]
    stride_size=config["Stride_Size"][0]
    feature_list=[i for i in config["Feature_List"]]
    return scaler,model,window_size,stride_size,feature_list

def get_len(data):
    len_pos=len(data[data["label"]==1].index)
    len_neg=len(data[data["label"]==0].index)
    return len_pos,len_neg

def get_pos_neg_data(data):
    pos_data_label=data[data["label"]==1]
    neg_data_label=data[data["label"]==0]

    pos_wrong_pred=pos_data_label[pos_data_label["pred_label"]==0]
    pos_true_pred = pos_data_label[pos_data_label["pred_label"] == 1]

    neg_wrong_pred=neg_data_label[neg_data_label["pred_label"]==1]
    neg_true_pred = neg_data_label[neg_data_label["pred_label"] == 0]

    return pos_data_label,neg_data_label,pos_wrong_pred,pos_true_pred,neg_wrong_pred,neg_true_pred

def check_missing_data(model_data_len_pos,model_data_len_neg):
    key=None
    if (model_data_len_pos != 0) and (model_data_len_neg != 0):
        key='all'
    elif (model_data_len_pos == 0) and (model_data_len_neg != 0):
        key='neg'
    elif (model_data_len_pos != 0) and (model_data_len_neg == 0):
        key='pos'
    elif (model_data_len_pos==16) and (model_data_len_neg==16):
        key='no_data'
    return key

def predict(model):
    prediction=model.predict(WindowedX)
    predict_labels=[0 if value[0]>value[1] else 1 for value in prediction]
    return predict_labels

def measure_perf(data,key):
    f1_score='no_data'
    recall_0='no_data'
    recall_1='no_data'
    acc='no_data'

    if key=='all':
        report=classification_report(data["label"],data["pred_label"],output_dict=True)
        f1_score=float("{:.3f}".format(report['macro avg']['f1-score']))
        recall_0=float("{:.3f}".format(recall_score(data["label"],data["pred_label"],pos_label=0)))
        recall_1=float("{:.3f}".format(recall_score(data["label"],data["pred_label"],pos_label=1)))
        acc=float("{:.3f}".format(report['accuracy']))
    elif key=='neg':
        recall_0=float("{:.3f}".format(recall_score(data["label"],data["pred_label"],pos_label=0)))
    elif key=='pos':
        recall_1=float("{:.3f}".format(recall_score(data["label"],data["pred_label"],pos_label=1)))

    return f1_score, recall_0, recall_1, acc


def plot(f1_score_radar, recall_0_radar, recall_1_radar, acc_radar,
                pos_data_label_radar, neg_data_label_radar, pos_wrong_pred_radar,
            pos_true_pred_radar, neg_wrong_pred_radar, neg_true_pred_radar,
         model_name,flight_name,model_result_save_path,model_data_len_pos,model_data_len_neg):
    if mode:
        radar_type='Echodyne'
    else:
        radar_type='Acar'
    fig,(ax1,ax2)=plt.subplots(1,2)
    fig.suptitle("Drone Predictions")

    ax1.scatter("pos_x","pos_y",data=pos_wrong_pred_radar, marker='*', c = "red" , label= 'Wrong Drone Prediction')
    ax1.scatter("pos_x","pos_y",data=pos_true_pred_radar, marker='*', c = "blue" , label= 'True Drone Prediction')
    ax1.set_xlabel(f"f1_score: {f1_score_radar}  recall_0: {recall_0_radar}  recall_1: {recall_1_radar}  acc: {acc_radar}")
    ax1.set_title(f"{radar_type} Tahmini")
    ax1.legend(loc='upper right')

    ax2.scatter("pos_x","pos_y",data=pos_wrong_pred, marker='*', c = "red" , label= f'Wrong Drone Prediction')
    ax2.scatter("pos_x","pos_y",data=pos_true_pred, marker='*', c = "blue" , label= f'True Drone Prediction')
    ax2.set_xlabel(f"f1_score: {f1_score}  recall_0: {recall_0}  recall_1: {recall_1}  acc: {acc} pos: {model_data_len_pos} neg: {model_data_len_neg}")
    ax2.set_title(str(model_name[1]))
    first_legend=plt.legend

    ax2.legend(loc='upper left')

    plt.savefig(str(model_result_save_path[0]) + "/" + str(model_name[1]) + "_" + str(
        flight_name[1]) + "_drone_grafik.png")
    print(str(model_name[1]))
    print(str(flight_name[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Negative Predictions')

    ax1.scatter("pos_x","pos_y",data=neg_true_pred_radar, marker='*', c = "red" , label='True Neg Pred')
    ax1.scatter("pos_x","pos_y",data=neg_wrong_pred_radar,marker='*', c = "blue" , label='Wrong Neg Pred')
    ax1.set_xlabel(f"f1_score: {f1_score_radar}  recall_0: {recall_0_radar}  recall_1: {recall_1_radar}  acc: {acc_radar}")
    ax1.set_title(f"{radar_type} Tahmini")
    ax1.legend(loc='upper right')
    ax1.legend()

    ax2.scatter("pos_x","pos_y",data=neg_true_pred,marker='*', c = "red" , label='True Neg Pred')
    ax2.scatter("pos_x", "pos_y", data=neg_wrong_pred, marker='*', c="blue", label='Wrong Neg Pred')
    ax2.set_xlabel(f"f1_score: {f1_score}  recall_0: {recall_0}  recall_1: {recall_1}  acc: {acc} pos: {model_data_len_pos} neg: {model_data_len_neg}")
    ax2.set_title(str(model_name[1]))
    ax2.legend(loc='upper right')

    plt.savefig(str(model_result_save_path[0]) + "/" + str(model_name[1]) + "_" + str(
        flight_name[1]) + "_d_negatif_grafik.png")


def echo_or_acar(mode):
    f1_score_radar,recall_0_radar,recall_1_radar,acc_radar=None,None,None,None
    if mode:
        echo_data_len_pos, echo_data_len_neg = get_len(echodyne_data)
        key_echo = check_missing_data(echo_data_len_pos, echo_data_len_neg)
        f1_score_radar,recall_0_radar,recall_1_radar,acc_radar = measure_perf(echodyne_data, key_echo)
        pos_data_label, neg_data_label, pos_wrong_pred, pos_true_pred, neg_wrong_pred, neg_true_pred=get_pos_neg_data(echodyne_data)
    else:
        acar_data_len_pos, acar_data_len_neg = get_len(acar_data)
        key_acar = check_missing_data(acar_data_len_pos, acar_data_len_neg)
        f1_score_radar,recall_0_radar,recall_1_radar,acc_radar = measure_perf(acar_data, key_acar)
        pos_data_label, neg_data_label, pos_wrong_pred, pos_true_pred, neg_wrong_pred, neg_true_pred=get_pos_neg_data(acar_data)

    return f1_score_radar,recall_0_radar,recall_1_radar,acc_radar,\
pos_data_label, neg_data_label, pos_wrong_pred, pos_true_pred, neg_wrong_pred, neg_true_pred


if __name__=="__main__":
    mode=0  #0 fusion, 1 echodyne
    model_path_list,data_path_list,model_result_save_path=define_paths(mode)
    for model_path in model_path_list:
        scaler,model,window_size,stride_size,feature_list=ihtar_model(model_path)
        model_shape=model.input_shape[1]
        for data_path in data_path_list:
            if mode:
                test_data,echodyne_data=parse_data(data_path,mode)  #echonedyne_data ve acar_data ismini data yapabilir miyiz?
            else:
                test_data,acar_data=parse_data(data_path,mode)

            WindowedX, WindowedY, PosXY= prepare_data(test_data,scaler,feature_list,window_size,stride_size,model_shape)
            pred_label=predict(model)
            pos_label=[i[-1] for i in PosXY]
            model_data=pd.DataFrame(pos_label,columns=("pos_x","pos_y","label"))
            model_data["pred_label"]=pred_label
            model_data_len_pos,model_data_len_neg=get_len(model_data)
            key=check_missing_data(model_data_len_pos,model_data_len_neg)
            f1_score, recall_0, recall_1, acc=measure_perf(model_data,key)
            pos_data_label,neg_data_label,pos_wrong_pred,pos_true_pred,neg_wrong_pred,neg_true_pred=get_pos_neg_data(model_data)

            model_name=os.path.split(model_path)
            flight_name=os.path.split(data_path)

            f1_score_radar, recall_0_radar, recall_1_radar, acc_radar,\
                pos_data_label_radar, neg_data_label_radar, pos_wrong_pred_radar, \
            pos_true_pred_radar, neg_wrong_pred_radar, neg_true_pred_radar=echo_or_acar(mode)

            plot(f1_score_radar, recall_0_radar, recall_1_radar, acc_radar,
                pos_data_label_radar, neg_data_label_radar, pos_wrong_pred_radar,
            pos_true_pred_radar, neg_wrong_pred_radar, neg_true_pred_radar,
                 model_name,flight_name,model_result_save_path,model_data_len_pos,model_data_len_neg)



































