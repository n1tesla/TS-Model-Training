import pandas as pd
import numpy as np
import glob 
import os
from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
import pickle
import joblib
from functions_list import *
import sys
import warnings
import matplotlib
matplotlib.use('Agg')


if not sys.warnoptions:
    warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)


model_path_root = r"C:\Users\tunahan.akyol\Desktop\lstm\flight_based_test\models_tobe_compared\model_tobe_compared"

data_path_root = r"C:\Users\tunahan.akyol\Desktop\lstm\flight_based_test\test_flights\test_veri_ortamı2"
data_path_list = []

for file in glob.glob(data_path_root + "/*"):
    data_path_list.append(file)


model_path_list = []
for file in glob.glob(model_path_root + "/*"):
    if file == model_path_root + '\\ModelResults':
        continue
    model_path_list.append(file)

model_result_save_path = glob.glob(model_path_root  + "/ModelR*")
model_name = []
F1_Score = []
Acc_score = []
flight_name = []

for model_path in model_path_list:
    model_config_path = glob.glob(model_path + "\\config.csv")
    scaler_path = glob.glob(model_path + "\\scaler*")
    config_file = pd.read_csv(model_config_path[0])
    window_size = config_file["Window_Size"][0]
    stride_size = config_file["Stride_Size"][0]
    feature_list = []
    for i in config_file["Feature_List"]:
        feature_list.append(i)
    scaler = joblib.load(str(scaler_path[0]))
    model =  tf.keras.models.load_model(model_path)
    #print(scaler, model)
    for data_list in data_path_list:
        # neg_path =  glob.glob(data_list + "\\extractedData_neg*")
        # pos_path = glob.glob(data_list + "\\extractedData_pos*")
        # data_pos = pd.read_csv(pos_path[0])
        # data_neg = pd.read_csv(neg_path[0])
        # all_data = pd.concat([data_pos, data_neg]).reset_index().drop(["index", ], axis = 1)
        all_data=pd.read_csv(data_list+"\\extractedData_All_Train.csv")
        # model_scaled_data = scaler.transform(np.array(temp_model_data))
        #overlap trackid
        all_data.loc[:, feature_list] = scaler.transform(all_data.loc[:, feature_list].values)
        WindowedX,WindowedY,PosXY=overlap_TrackID(all_data,'id',window_size,stride_size,feature_list)

        # model_input_data = extract_window(temp_model_data, window_size, stride_size)
        WindowedX=WindowedX.reshape((WindowedX.shape[0], len(feature_list), window_size))
        prediction = model.predict(WindowedX)

        pred_label = []

        for i,value in enumerate(prediction):
            if value[0]>value[1]:
                probability ='0'
                pred_label.append(probability)
            else:
                probability='1'
                pred_label.append(probability)
            # probility = prediction[i].mean()
            # if probility >= 0.5:
            #     pred_label.append("1")
            # else:
            #     pred_label.append("0")

        ecodyne_result_pred = []
        for prob in all_data["probUAV"]:
            if prob >= 0.5:
                ecodyne_result_pred.append("1")
            else:
                ecodyne_result_pred.append("0")

        
        ecodyne_result = all_data[["pos_x", "pos_y", "label"]]
        ecodyne_result["pred_label"] = ecodyne_result_pred
        ecodyne_result["ground_truth"] = ecodyne_result["label"]

        temp_all_data_locations = all_data[["pos_x", "pos_y", "label"]]

        index_list_temp = temp_all_data_locations.index
        
        index_list = []
        stride_size_counter = stride_size

        model_result = []
        for i in PosXY:
            temp_row = i[-1]
            model_result.append(temp_row)

        model_result = pd.DataFrame(model_result, columns=("pos_x", "pos_y", "label"))
        model_result["label"] = model_result["label"].astype(int)
        model_result["label"] = model_result["label"].astype(str)
        model_result["pred_label"] = pred_label
# # [-0.63158934, -0.53884449,  1.
#         for i,value in enumerate(index_list_temp):
#             if value == window_size - 1:
#                 index_list.append(i)
#             elif value == (window_size -1 ) + stride_size_counter:
#                 index_list.append(i)
#                 stride_size_counter += stride_size

        #print(index_list, len(index_list))


        #print(len(prediction), len(index_list))

        #ecodyne_result = ecodyne_result.loc[index_list]
        # all_data_locations= temp_all_data_locations.loc[index_list]
        # all_data_locations["pred_label"] = pred_label

        #print(all_data_locations, ecodyne_result)

        # model_result = all_data_locations


        #print(ecodyne_result, model_result)

        from sklearn.metrics import f1_score
        from sklearn.metrics import classification_report

        #print(len(model_result["label"]), len(model_result["pred_label"]))
        
        # Bu döngülerden kurtulanacak!
        label_list_eco = []
        for i in ecodyne_result["label"]:
            if i == 1:
                label_list_eco.append("1")
            elif i == 0:
                label_list_eco.append("0")

        # label_list_model = []
        # for i in model_result["label"]:
        #     if i == 1:
        #         label_list_model.append("1")
        #     elif i == 0:
        #         label_list_model.append("0")

        ecodyne_result["label"] = label_list_eco
        # model_result["label"] = label_list_model


        eco_f1 = f1_score(ecodyne_result["label"], ecodyne_result["pred_label"], average='macro')


        model_f1 =  f1_score(model_result["label"], model_result["pred_label"], average='macro')

        eco_class_report = classification_report(ecodyne_result["label"], ecodyne_result["pred_label"], output_dict=True)
        eco_recall_0 = eco_class_report["0"]["recall"]
        eco_recall_1 = eco_class_report["1"]["recall"]


        model_class_report =  classification_report(model_result["label"], model_result["pred_label"], output_dict=True)
        model_recall_0 = model_class_report["0"]["recall"]
        # model_recall_1 = model_class_report["1"]["recall"]

        ecodyne_result_neg = ecodyne_result[ecodyne_result["label"] == "0"]
        ecodyne_result_pos = ecodyne_result[ecodyne_result["label"] == "1"]

        model_result_neg = model_result[model_result["label"] == "0"]
        model_result_pos = model_result[model_result["label"] == "1"]

        #print(eco_f1, model_f1)
        

        eco_f1_3f = float("{:.4f}".format(eco_f1))
        eco_recall_0_3f = float("{:.4f}".format(eco_recall_0))
        eco_recall_1_3f = float("{:.4f}".format(eco_recall_1))

        model_f1_3f = float("{:.4f}".format(model_f1))
        model_recall_0_3f = float("{:.4f}".format(model_recall_0))
        model_recall_1_3f = float("{:.4f}".format(model_recall_1))


        model_name_tail = os.path.split(model_path)
        flight_name_tail = os.path.split(data_list)

        model_name.append(str(model_name_tail[1]))
        F1_Score.append(model_f1_3f)
        flight_name.append(str(flight_name_tail[1]))


        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Drone İçin Tahmin Grafikleri') 


        ax1.scatter( "pos_x", "pos_y", data= ecodyne_result_pos[ecodyne_result_pos["pred_label"] == "0"], marker='*', c = "red" , label= 'Drone Hatalı')
        ax1.scatter( "pos_x", "pos_y", data= ecodyne_result_pos[ecodyne_result_pos["pred_label"] == "1"], marker='*', c = "blue" , label='Drone Dogru')
        ax1.set_xlabel("F1 Score:" + str(eco_f1_3f) + "  Recall_0: " + str(eco_recall_0_3f)+ "  Recall_1: " + str(eco_recall_1_3f), fontsize=10)
        ax1.set_title("Echodyne  Tahmini")
        ax1.legend(loc= "upper right")

        
        ax2.scatter( "pos_x", "pos_y", data=model_result_pos[model_result_pos["pred_label"] == "0"], marker='*', c = "red" , label= 'Drone Hatalı')
        ax2.scatter( "pos_x", "pos_y", data=model_result_pos[model_result_pos["pred_label"] == "1"], marker='*', c = "blue" , label= 'Drone Dogru')
        ax2.set_title(str(model_name_tail[1]))
        ax2.set_xlabel("F1 Score:" + str(model_f1_3f) + "  Recall_0: " + str(model_recall_0_3f)+ "  Recall_1: " + str(model_recall_1_3f), fontsize=10)

        ax2.legend(loc= "upper right")

        plt.savefig(str(model_result_save_path[0]) + "/" + str(model_name_tail[1]) + "_"+ str(flight_name_tail[1]) +"_drone_grafik.png")
        
        print(str(model_name_tail[1]))
        print(str(flight_name_tail[1]))

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Bilinmeyen İçin Tahmin Grafikleri') 


        ax1.scatter( "pos_x", "pos_y", data= ecodyne_result_neg[ecodyne_result_neg["pred_label"] == "0"], marker='*', c = "red" , label='Negatif Dogru')
        ax1.scatter( "pos_x", "pos_y", data= ecodyne_result_neg[ecodyne_result_neg["pred_label"] == "1"], marker='*', c = "blue" , label='Negatif Hatalı')
        ax1.set_xlabel("F1 Score:" + str(eco_f1_3f) + "  Recall_0: " + str(eco_recall_0_3f)+ "  Recall_1: " + str(eco_recall_1_3f), fontsize=10)
        ax1.set_title("Echodyne  Tahmini")
        ax1.legend(loc= "upper right")

        
        ax2.scatter( "pos_x", "pos_y", data=model_result_neg[model_result_neg["pred_label"] == "0"], marker='*', c = "red" , label='Negatif Dogru')
        ax2.scatter( "pos_x", "pos_y", data=model_result_neg[model_result_neg["pred_label"] == "1"], marker='*', c = "blue" , label='Negatif Hatalı')
        ax2.set_title(str(model_name_tail[1]))
        ax2.set_xlabel("F1 Score:" + str(model_f1_3f) + "  Recall_0: " + str(model_recall_0_3f)+ "  Recall_1: " + str(model_recall_1_3f), fontsize=10)
        
        ax2.legend(loc= "upper right")

        plt.savefig(str(model_result_save_path[0]) + "/" + str(model_name_tail[1]) + "_"+ str(flight_name_tail[1]) +"_d_negatif_grafik.png")

model_name = pd.DataFrame(model_name, columns=['model_name'])
F1_Score = pd.DataFrame(F1_Score, columns=['F1_Score'])
flight_name = pd.DataFrame(flight_name, columns=['flight_name'])

        

model_result_all_in_one = pd.concat([model_name, F1_Score, flight_name], ignore_index=True, axis=1)
model_result_all_in_one.columns = ['model_name', 'F1_Score', 'flight_name']

model_result_all_in_one = model_result_all_in_one.sort_values(["flight_name"], ascending=True)

model_result_all_in_one.to_csv(str(model_result_save_path[0]) + "/" + "models_result.csv" , index=False)  
     

    



