import pandas as pd
import os
from data.data_preparation import CROSS_VAL_SPLIT,PREPARATION
import datetime
from joblib import dump
# from data.data_augmentation import AUGMENTATION
import ihtar_utils

if __name__ == '__main__':

    mode=0   #mode=0 fusion,mode=1 echodyne,
    framework=1 #1 tensorflow shape, 0 pytorch shape
    cwd = os.getcwd()
    if mode:
        dataset_path='data\\ecodyne'
        input_path=cwd+"\\"+dataset_path
        feature_columns = ["elevation", "vel", "acc", "range", "radialVel", "rcs"]
    else:
        dataset_path='data\\fusion'
        input_path=cwd+"\\"+dataset_path
        feature_columns = ["elevation", "vel", "acc", "range","radialVel","rcs_dsbm"]
    csv_number = len(os.listdir(input_path))

    #PREPARE DATA
    step_size = 2 #step size drone uçuşuna ait verinin train,val ve teste dağıtılırken atacağı adım sayısı. 2 den fazlası olması durumunda bazen test için veri kalmıyor.
    window_size, stride = 16,4
    shuffle_size = 50000  # shuffle size, toplam veri sayısından daha fazla olmalı. Tüm verinin karışabilmesi için.
    batch_size_list = [256] #denenmesini istediğiniz batch_size değerlerini girin.

    df_all_results=pd.DataFrame([])
    label_features = ["label", "id","pos_x","pos_y","probUAV"]
    architecture = 'fcn'  # lstm, fcn, mlstm-fcn, malstm-fcn,
    observation='' #test sonucunda gözlemlenmek istenen şeyin adını yaz. Öne çıkan baskın özellik. Örn: feature, batch_size, dataset_ratio
    observation_name=f'{architecture}{csv_number}w{window_size}s{stride}{observation}' #csv_sayısı, window_size ve stride göre otomatik path oluşturmak için.
    observation_path = cwd + "\\saved_models\\" + observation_name + "\\"
    lr_list = [5e-6,1e-5,5e-5,1e-4] #modeli farklı öğrenme hızlarıyla eğiterek en iyi konfigürasyon bulunması amaçlanıyor.
    loss_patience_list = [16,12,10,8] # stop earlynın içinde yer alan patience değeri. Öğrenme hızı ile ters orantılı.
    max_trials=30
    number_of_models_to_save=20
    #DATA AUGMENTATION CONFIG
    mask_features=["rcs"] # seçilen feature veriseti hazırlanırken rastgele bir şekilde kapatılacak (değeri 0 yapılacak)
    mask_ratio=0.1 #0.5 ten küçük olması önerilir.
    padding_threshold=12 # 12 VE 12 den büyük ID uzunluklarını window_size uzunluğuna tamamlar.
    augmentation,padding,masking,shuffling=0,0,0,0
    pos_train_dts_ratio_l=[0.6,0.7,0.8]
    neg_train_dts_ratio_l=[0.6,0.7,0.8]
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ihtar_utils.make_dir(observation_path)
    time_path = observation_path + start_time
    ihtar_utils.make_dir(time_path)
    for index_pos,pos_train_dts_ratio in enumerate(pos_train_dts_ratio_l):
        for index_neg,neg_train_dts_ratio in enumerate(neg_train_dts_ratio_l):
            for k,batch_size in enumerate(batch_size_list):
                for l,lr in enumerate(lr_list):
                    patience=loss_patience_list[l]
                    run_path=time_path+f"\\bs{batch_size}_{pos_train_dts_ratio}_{neg_train_dts_ratio}_lr{lr}"
                    ihtar_utils.make_dir(run_path)
                    cross_val=CROSS_VAL_SPLIT(feature_columns, pos_train_dts_ratio,neg_train_dts_ratio,
                                              window_size, stride, run_path,csv_number,padding,padding_threshold,dataset_path)
                    df_train,df_val=cross_val.split()
                    ts=PREPARATION(feature_columns,window_size,stride,df_train,df_val,True,mode,framework)
                    dataset_dict,scaler=ts.create_dataset()
                    dump(scaler, open(os.path.join(run_path+"\\data", 'scaler.bin'), 'wb')) #save scaler.
                    X_train, y_train = dataset_dict['df_train'][0], dataset_dict['df_train'][1]
                    X_valid, y_valid = dataset_dict['df_val'][0], dataset_dict['df_val'][1]

                    from train import TRAIN_FCN
                    train = TRAIN_FCN(feature_columns, label_features, window_size, stride, X_train, y_train, X_valid,
                                      y_valid,scaler, batch_size, start_time, run_path, observation_name,
                                      lr, patience, max_trials, number_of_models_to_save, architecture,dataset_dict,pos_train_dts_ratio,neg_train_dts_ratio)
                    df_batch_results=train.BayesianOptimization_FCN()
                    df_batch_results.to_csv(run_path + "\\test_results.csv")  # batch test sonuçlarını kaydet
                    df_all_results=pd.concat([df_all_results,df_batch_results[:3]])
    df_all_results=df_all_results.sort_values(by=["batch"],ascending=False)
    df_all_results.to_csv(time_path+"\\best_of_all.csv")



