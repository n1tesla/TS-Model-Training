from tsai_ihtar.all import *
from data.data_preparation import CROSS_VAL_SPLIT,PREPARATION
import datetime
from joblib import dump
import ihtar_utils
from sklearn.metrics import *
from test import test_torch_models
from search_hyper import *
computer_setup()

if __name__=='__main__':
    all_archs_names = ['FCN', 'FCNPlus', 'InceptionTime', 'InceptionTimePlus', 'InCoordTime', 'XCoordTime',
                       'InceptionTimePlus17x17', 'InceptionTimePlus32x32',
                       'InceptionTimePlus47x47', 'InceptionTimePlus62x62', 'InceptionTimeXLPlus',
                       'MultiInceptionTimePlus', 'MiniRocketClassifier',
                       'MiniRocketRegressor', 'MiniRocketVotingClassifier', 'MiniRocketVotingRegressor',
                       'MiniRocketFeaturesPlus', 'MiniRocketPlus',
                       'MiniRocketHead', 'InceptionRocketFeaturesPlus', 'InceptionRocketPlus', 'MLP', 'gMLP',
                       'MultiInputNet', 'OmniScaleCNN', 'RNN', 'LSTM', 'GRU',
                       'RNNPlus', 'LSTMPlus', 'GRUPlus', 'RNN_FCN', 'LSTM_FCN', 'GRU_FCN', 'MRNN_FCN', 'MLSTM_FCN',
                       'MGRU_FCN', 'ROCKET', 'RocketClassifier',
                       'RocketRegressor', 'ResCNN', 'ResNet', 'ResNetPlus', 'TCN', 'TSPerceiver', 'TST', 'TSTPlus',
                       'MultiTSTPlus', 'TSiTPlus', 'TSiTPlus',
                       'TabFusionTransformer', 'TSTabFusionTransformer', 'TabModel', 'TabTransformer',
                       'GatedTabTransformer', 'TransformerModel', 'XCM', 'XCMPlus', 'xresnet1d18',
                       'xresnet1d34', 'xresnet1d50', 'xresnet1d101', 'xresnet1d152', 'xresnet1d18_deep',
                       'xresnet1d34_deep', 'xresnet1d50_deep',
                       'xresnet1d18_deeper', 'xresnet1d34_deeper', 'xresnet1d50_deeper', 'XResNet1dPlus',
                       'xresnet1d18plus', 'xresnet1d34plus',
                       'xresnet1d50plus', 'xresnet1d101plus', 'xresnet1d152plus', 'xresnet1d18_deepplus',
                       'xresnet1d34_deepplus', 'xresnet1d50_deepplus',
                       'xresnet1d18_deeperplus', 'xresnet1d34_deeperplus', 'xresnet1d50_deeperplus', 'XceptionTime',
                       'XceptionTimePlus', 'mWDN',
                       'TSSequencer', 'TSSequencerPlus']
    mode=0   #mode=0 fusion,mode=1 ecodyne,
    framework=0  # 0 pytorch, 1 tensorflow
    cwd = os.getcwd()
    if mode:
        dataset_path='data\\ecodyne'
        input_path=cwd+"\\"+dataset_path
        feature_columns = ["elevation", "vel", "acc", "range", "radialVel", "rcs"]
    else:
        dataset_path='data\\fusion'
        input_path=cwd+"\\"+dataset_path
        feature_columns = ["elevation", "vel", "acc", "range", "radialVel", "rcs_dsbm"]

    csv_number=len(os.listdir(input_path))
    window_size,stride=16,4
    batch_size = 64

    arch_name = 'InceptionTime'
    observation_name=f'{arch_name}{csv_number}w{window_size}s{stride}'
    observation_path =cwd+"\\saved_models\\" + observation_name + "\\"

    patience=5
    max_trials=30
    epoch=500
    feature_importance=True
    step_importance=True
    # torch.cuda.set_per_process_memory_fraction(0.5)
    pos_train_dts_ratio_l=[0.74,0.7]
    neg_train_dts_ratio_l=[0.74,0.7]
    padding_threshold=12 # 12 VE 12 den büyük ID uzunluklarını window_size uzunluğuna tamamlar.
    augmentation,padding,masking,shuffling=0,0,0,0
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    ihtar_utils.make_dir(observation_path)
    time_path = observation_path + start_time
    ihtar_utils.make_dir(time_path)

    df_all_results=pd.DataFrame()
    # best_results_pd=pd.DataFrame(data={'model_name':None,'bs':batch_size,'ws':window_size,'stride':stride})
    for index_pos,pos_train_dts_ratio in enumerate(pos_train_dts_ratio_l):
        for index_neg,neg_train_dts_ratio in enumerate(neg_train_dts_ratio_l):
            cross_val=CROSS_VAL_SPLIT(feature_columns,pos_train_dts_ratio,neg_train_dts_ratio,window_size,stride,
                                      time_path,csv_number,padding,padding_threshold,dataset_path)
            df_train,df_val=cross_val.split()

            ts=PREPARATION(feature_columns,window_size,stride,df_train,df_val,True,mode,framework)
            #AUGMENTATION part will be here
            dataset_dict,scaler=ts.create_dataset()
            dump(scaler, open(os.path.join(time_path + "\\data", 'scaler.bin'), 'wb'))  # save scaler.
            X_train, y_train = dataset_dict['df_train'][0], dataset_dict['df_train'][1]
            X_valid, y_valid = dataset_dict['df_val'][0], dataset_dict['df_val'][1]
            X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
            tfms = [None, TSClassification()]  # [Categorize()] da çalıştı
            dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
            dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=batch_size, inplace=True)

            #define paths
            run_path=time_path+f"\\bs{batch_size}_{pos_train_dts_ratio}_{neg_train_dts_ratio}"
            ihtar_utils.make_dir(run_path)
            models_dir = os.path.join(run_path, 'models')
            plots_path=os.path.join(run_path,'plots')
            ihtar_utils.make_dir(models_dir)
            ihtar_utils.make_dir(plots_path)
            #iterasyon yapılacak kısım. optuna buraya gelecek.

            #optuna
            tuner=TUNER(X,y,splits,epoch,batch_size,patience,arch_name)
            objective=tuner.get_objective()
            study = optuna.create_study(direction='minimize')
            study.optimize(objective,n_trials=max_trials)

            fig_hist=optuna.visualization.plot_optimization_history(study)
            fig_hist.write_image(file=plots_path+"\\optimization_history.png",format='png')
            fig_imp=optuna.visualization.plot_param_importances(study)
            fig_imp.write_image(file=plots_path+"\\param_imp.png",format='png')
            fig_slice=optuna.visualization.plot_slice(study)
            fig_slice.write_image(file=plots_path+"\\plot_slice.png",format='png')
            fig_coor=optuna.visualization.plot_parallel_coordinate(study)
            fig_coor.write_image(file=plots_path+"\\parallel_coor.png",format='png')

            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            df_batch_results=pd.DataFrame()
            for index,trial in enumerate(study.get_trials()):
                #retrain the model with best parameter
                if trial.values[0]>0.3:
                    continue
                arch_config=tuner.get_arch_config(trial.params)

                learning_rate = trial.params['learning_rate']
                learn = TSClassifier(X, y, splits=splits, bs=batch_size,
                                     arch=arch_name,
                                     arch_config=arch_config,
                                     metrics=accuracy)
                learn.fit_one_cycle(epoch, lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=patience),
                                    ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])
                str_lr=str(learning_rate)[:3]+str(learning_rate)[-4:]

                num_run_model = f"{observation_name}_{batch_size}_{str_lr}_{start_time[-4:]}_{index}"  # eşsiz model isimleri için dinamik yapı.
                model_path = os.path.join(models_dir, num_run_model)
                ihtar_utils.make_dir(model_path)
                # summary_path = model_path + "/summary.txt"  # model özetini txte kaydet.
                # with open(os.path.abspath(summary_path), "w+") as f:
                #     with redirect_stdout(f):
                #         print("Features:", feature_columns)
                #         print(f"Hyperparameters:", trial.params)
                # learn = Learner(dls, model, metrics=accuracy,model_dir=run_path)
                # learn.fit_one_cycle(15, lr_max=1e-5)
                if feature_importance:
                    learn.feature_importance(X,y,feature_names=feature_columns,save_path=plots_path)
                    feature_importance=True
                if step_importance:
                    learn.step_importance(X,y,save_path=plots_path)
                    step_importance=False
                learn.recorder.plot_metrics(save_path=model_path)
                learn.save_all(model_path)

                dls=learn.dls
                valid_dl=dls.valid
                valid_probas,valid_targets,valid_preds=learn.get_preds(dl=valid_dl,with_decoded=True)
                # print((valid_targets==valid_preds).float().mean())

                hp_config={}
                hp_config['bs']=batch_size
                hp_config['ws']=window_size
                hp_config['stride']=stride
                # hp_config['ds_ratio']=[pos_train_dts_ratio,neg_train_dts_ratio]
                hp_config['lr_rate']=learning_rate
                hp_config.update(arch_config)

                df_result=test_torch_models(learn, valid_dl, dataset_dict, index,hp_config)
                df_batch_results=pd.concat([df_batch_results,df_result])

                learn.show_results()
                plt.savefig(model_path+"\\results.png")
                learn.show_probas(save_path=model_path)
                del learn
            if mode:
                df_batch_results = df_batch_results.sort_values(by=["batch"], ascending=False)
            df_all_results=pd.concat([df_all_results,df_batch_results[:1]])
            df_batch_results.to_csv(models_dir + "\\test_results.csv")
            #test model
            model_param = pd.DataFrame()
            model_param.loc[:, 'Feature_List'] = feature_columns
            model_param['Window_Size'] = window_size
            model_param['Stride_Size'] = stride
            model_param['scale_'] = scaler.scale_  # C++ entegrasyon kodu için eğitimde kullanılan özniteliklerin standard sapmayı ve ortalamayı kaydet.
            model_param['mean_'] = scaler.mean_
            model_param.to_csv(model_path + "\\config.csv")
    if mode:
        df_all_results=df_all_results.sort_values(by=["batch"],ascending=False)
    df_all_results.to_csv(time_path+"\\best_of_all.csv")























