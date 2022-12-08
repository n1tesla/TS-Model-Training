import os
import numpy as np
import pandas as pd
from sklearn import metrics
from flight_based_test.functions_list import *

def test_torch_models(learn,valid_dl,dataset_dict,index,hp_config):
    df_result = pd.DataFrame([])
    for k,v in dataset_dict.items():
        #TODO MODEL ISMI EKLENCEK
        if k in ['df_train','df_val']:
            continue
        x_test,y_test=v[0],v[1]
        test_ds=valid_dl.dataset.add_test(x_test,y_test)
        test_dl=valid_dl.new(test_ds)
        test_probas,test_targets,test_preds=learn.get_preds(dl=test_dl,with_decoded=True,save_preds=None,save_targs=None)
        report=metrics.classification_report(test_targets,test_preds,output_dict=True,zero_division=0)
        accuracy=report['accuracy']
        df_result['model_no']=index
        df_result.loc[:, k] = [accuracy]
        print(f"model got {accuracy} accuracy from {k} test set")

    for k2, v2 in hp_config.items():
        # if k2=='ds_ratio':
        #     df_result.loc[:,k2]=v2
        df_result.loc[:, k2] = [v2]
    return df_result


def tensorflow_models(model,dataset_dict,model_index,hp_config):
    df_result=pd.DataFrame([])
    for k,v in dataset_dict.items():
        if k in ['df_train','df_val']:
            continue
        prediction_list,ground_truth = [],[]
        x_test, y_test, posxy_label = v[0], v[1], v[2]
        score=model.evaluate(x_test,y_test) #score[0] loss, score[1] accuracy
        print(f"loss: {score[0]} acc: {score[1]}")
        accuracy=score[1]

        df_result['model_no']=model_index
        df_result.loc[:,k]=[accuracy]
        print(f"model got {accuracy} accuracy from {k} test set")
    for k2, v2 in hp_config.items():
        df_result.loc[:, k2] = [v2]
    return df_result


def pos_batch_test_fcn(batch_test_X,batch_test_y,model):
    batch_tahmin_listesi = []
    batch_ground_truth = []

  # cross-val testinden iyi sonuç alan modelleri batch teste sokuyoruz.
    batch_predictions=model.predict(np.reshape(batch_test_X,batch_test_X.shape))
    for index, result in enumerate(batch_predictions):
        if result[0] >result[1]:
            batch_prediction = 0
        else:
            batch_prediction = 1
        batch_tahmin_listesi.append(batch_prediction)
        batch_ground_truth.append(batch_test_y[index])  # tahmin edilen verinin gerçek değerine bak
    batch_report = metrics.classification_report(batch_ground_truth, batch_tahmin_listesi, output_dict=True,zero_division=0)

    macro_f1_score=batch_report["macro avg"]["f1-score"]
    recall_1 = batch_report["1.0"]["recall"]

    return macro_f1_score,recall_1


def neg_batch_test_fcn(batch_test_X,batch_test_y,model):
    batch_tahmin_listesi = []
    batch_ground_truth = []

  # cross-val testinden iyi sonuç alan modelleri batch teste sokuyoruz.
    batch_predictions=model.predict(np.reshape(batch_test_X,batch_test_X.shape))
    for index, result in enumerate(batch_predictions):
        if result[0] >result[1]:
            batch_prediction = 0
        else:
            batch_prediction = 1
        batch_tahmin_listesi.append(batch_prediction)
        batch_ground_truth.append(batch_test_y[index])  # tahmin edilen verinin gerçek değerine bak
    batch_report = metrics.classification_report(batch_ground_truth, batch_tahmin_listesi, output_dict=True,zero_division=0)

    recall_0 = batch_report["0.0"]["recall"]

    return recall_0

def neg_batch_test(data_path,model,window_size,stride_size,feature_columns,scaler):
    prediction_list=[]
    ground_truth=[]
    df_neg = pd.read_csv(data_path)
    df_neg.loc[:, feature_columns] = scaler.transform(df_neg.loc[:, feature_columns].values)
    test_X, test_y = overlap_TrackID(df_neg, 'id', window_size, stride_size, feature_columns)
    predictions=model.predict(np.reshape(test_X,test_X.shape))
    for index,result in enumerate(predictions):
        if result[0]>result[1]:
            prediction=0
        else:
            prediction=1
        prediction_list.append(prediction)
        ground_truth.append(test_y[index])
    report=metrics.classification_report(ground_truth,prediction_list,output_dict=True,zero_division=0)
    recall_0 = report["0.0"]["recall"]
    return recall_0
    #TODO: en iyi batchleri neg teste sok


def test_synthetic(cwd,model,window_size,stride,feature_columns,scaler,model_path):
    df_synthetic = pd.DataFrame([])
    record_time_list=[]
    recall_1_list,recall_0_list,precision_0_list,precision_1_list=[],[],[],[]
    speed_list=[]
    for speed in os.listdir(cwd + '\synthetic_test'):
        speed_path = cwd + '\\synthetic_test' + "\\" + speed
        for record_time in os.listdir(speed_path):
            speed_list.append(speed)
            record_time_path = speed_path + "\\" + record_time
            each_synthetic = pd.read_csv(record_time_path + "\\extractedData_Drone_DT.csv")
            each_synthetic.loc[:,feature_columns]=scaler.transform(each_synthetic.loc[:,feature_columns].values)
            test_X,test_y=overlap_TrackID(each_synthetic,'id',window_size,stride,feature_columns)
            pred_label = []
            ground_truth=[]
            predictions=model.predict(np.reshape(test_X,test_X.shape))
            for index,result in enumerate(predictions):
                if result[0]>result[1]: #softmax aktivasyon fonksiyonu 2 çıktılı
                    prediction=0
                else:
                    prediction=1
                pred_label.append(prediction)
                ground_truth.append(test_y[index])
            synthetic_report=metrics.classification_report(ground_truth,pred_label,output_dict=True,zero_division=0)
            model_recall_1=synthetic_report["1.0"]["recall"]
            recall_1_list.append(model_recall_1)
            record_time_list.append(record_time)

    df_synthetic['flight_name']=record_time_list
    df_synthetic["recall_1"]=recall_1_list
    df_synthetic["speed"]=speed_list
    df_synthetic.to_csv(model_path+"\\synthetic_result.csv")



def overlap_TrackID(df,based_id,window_size,stride,feature_columns):
    WindowedX, WindowedY = [], []
    UniqueSplitIds = df[based_id].unique()
    for unique in UniqueSplitIds:
        Split_Data = df[based_id].isin([unique])
        SplitDataX = df.loc[Split_Data, feature_columns].to_numpy()
        SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
        #print(f"SplitDataX: {SplitDataX.shape[0]},SplitDataY: {SplitDataY[0]},split:{unique}")
        if SplitDataX.shape[0] >= window_size:
            WindowedX.append(extract_window(SplitDataX, window_size,stride))
            WindowedY.append(extract_window(SplitDataY,  window_size,stride))
    WindowedX = np.concatenate(WindowedX, axis=0)
    WindowedY = np.concatenate(WindowedY, axis=0)
    CNN_y = np.array([i.mean() for i in WindowedY])
    return WindowedX, CNN_y

def extract_window(arr, size, stride):
    examples = []
    min_len = size - 1
    max_len = len(arr) - size
    for i in range(0, max_len + 1, stride):
        example = arr[i:size + i]
        examples.append(np.expand_dims(example, 0))
    return np.vstack(examples)


