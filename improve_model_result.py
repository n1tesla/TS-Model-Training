import os
import pandas as pd
from sklearn import metrics
from flight_based_test.functions_list import *
import tensorflow as tf
feature_columns = ["elevation","vel", "acc","range","angularVel","radialVel","rcs"]
def overlap_TrackID(df,based_id,window_size,stride,feature_columns):
    WindowedX, WindowedY,PosXY,SplitId = [],[],[],[]
    UniqueSplitIds = df[based_id].unique()
    for unique in UniqueSplitIds:
        Split_Data = df[based_id].isin([unique])

        SplitDataPosXY=df.loc[Split_Data,['pos_x','pos_y','label']].to_numpy()
        Split_ID_data=df.loc[Split_Data,['id']].to_numpy()
        SplitDataX = df.loc[Split_Data, feature_columns].to_numpy()
        SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
        #print(f"SplitDataX: {SplitDataX.shape[0]},SplitDataY: {SplitDataY[0]},split:{unique}")
        if SplitDataX.shape[0] >= window_size:
            WindowedX.append(extract_window(SplitDataX, window_size,stride))
            WindowedY.append(extract_window(SplitDataY,  window_size,stride))
            PosXY.append(extract_window(SplitDataPosXY,window_size,stride))
            SplitId.append(extract_window(Split_ID_data,window_size,stride))
    WindowedX = np.concatenate(WindowedX, axis=0)
    WindowedY = np.concatenate(WindowedY, axis=0)
    PosXY=np.concatenate(PosXY,axis=0)
    SplitId=np.concatenate(SplitId,axis=0)

    CNN_y = np.array([i.mean() for i in WindowedY])
    return WindowedX, CNN_y,PosXY,SplitId

def extract_window(arr, size, stride):
    examples = []
    min_len = size - 1
    max_len = len(arr) - size
    for i in range(0, max_len + 1, stride):
        example = arr[i:size + i]
        examples.append(np.expand_dims(example, 0))
    return np.vstack(examples)

def test(cwd,window_size,stride,feature_columns,scaler,model_path):

    data_path=r'C:\Users\tunahan.akyol\Desktop\Export_20220216_ValData\test_data'
    model=tf.keras.models.load_model(model_path)
    for i,record_time in enumerate(os.listdir(data_path)):

        df_outputs = pd.DataFrame([])
        record_path=data_path+"\\"+record_time

        df_test=pd.read_csv(record_path+"\\extractedData_All_Test.csv")
        df_test.loc[:, feature_columns] = scaler.transform(df_test.loc[:, feature_columns].values)
        test_X, test_y,PosXY,SplitId = overlap_TrackID(df_test, 'id', window_size, stride, feature_columns)

        pred_label=[]
        ground_truth=[]
        score_0=[]
        score_1=[]
        positions_x=[]
        positions_y=[]
        split_id=[]
        predictions=model.predict(np.reshape(test_X,test_X.shape))
        for index,result in enumerate(predictions):
            if result[0]>result[1]:
                prediction=0
            else:
                prediction=1
            pred_label.append(prediction)
            ground_truth.append(test_y[index])
            positions_x.append(PosXY[index][-1][0])
            positions_y.append(PosXY[index][-1][1])
            split_id.append(SplitId[index][-1][0])
            score_0.append(result[0])
            score_1.append(result[1])

        report=metrics.classification_report(ground_truth,pred_label,output_dict=True,zero_division=0)
        df_outputs['score_0']=score_0
        df_outputs['score_1']=score_1
        df_outputs["pos_x"]=positions_x
        df_outputs["pos_y"]=positions_y
        df_outputs['id']=split_id
        df_outputs['label']=ground_truth
        df_outputs['prediction']=pred_label

        recall_0 = report["0.0"]["recall"]
        recall_1=report["1.0"]["recall"]
        macro_f1_score=report["macro avg"]["f1-score"]
        df_outputs['recall_0']=np.ones(len(score_0))*recall_0
        df_outputs['recall_1']=np.ones(len(score_0))*recall_1
        df_outputs['f1_score']=np.ones(len(score_0))*macro_f1_score
        df_outputs.to_csv(record_path+f"\\outputs.csv")
    return True
model_path=r'C:\Users\tunahan.akyol\Desktop\24w16s4elevelaccranangradrcs_5120.0001_27'

import joblib

scaler = joblib.load(model_path+"\\scaler.bin")
test(os.getcwd(),16,4,feature_columns,scaler,model_path)

