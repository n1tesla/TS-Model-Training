import datetime
import os
import time
import pandas as pd
from contextlib import redirect_stdout
from joblib import dump
from save_plots import save_fig
import plot_metrics
import numpy as np
import test
from sklearn import metrics
import math
import tensorflow as tf
from custom_metrics import *
from tensorflow.python.keras.callbacks import TensorBoard
os.environ["KERAS_BACKEND"] = "theano"
import shap
import ihtar_utils
import keras_tuner
import configparser
"""This is a module docstring, shown when you use help() on a module"""
class TRAIN_FCN(object):
    """Help for train FCN"""
    def __init__(self,feature_columns, label_features, window_size, stride,X_train,y_train,X_val,y_val,
                 scaler, batch_size, start_time, run_path, observation_name,
                            lr,patience,max_trials,number_of_models_to_save,architecture,dataset_dict,pos_ds_ratio,neg_ds_ratio):

        self.window_size=window_size
        self.stride_size=stride
        self.feature_columns=feature_columns
        self.label_features=label_features
        self.X_train=X_train
        self.y_train=y_train
        self.X_val=X_val
        self.y_val=y_val
        self.scaler = scaler
        self.cwd=os.getcwd()
        self.batch_size=batch_size
        self.start_time=start_time
        self.run_path=run_path
        self.observation_name=observation_name
        self.lr=lr
        self.patience=patience
        self.max_trials = max_trials
        self.number_of_models_to_save = number_of_models_to_save
        self.architecture=architecture
        self.nb_classes=2
        self.dataset_dict=dataset_dict
        self.pos_ds_ratio=pos_ds_ratio
        self.neg_ds_ratio=neg_ds_ratio
    def BayesianOptimization_FCN(self): #Bayessian algoritması val_loss'u düşürmeye yönelik adımlar atıyor. Bir önceki searchten dersler alarak, her searchte daha iyi parametreler seçiyor.
        """help for bayesianoptimization"""
        if self.architecture=='fcn':
            from models.fcn import FCN as CNN_ARC
            # from models.fcn import stable_fcn as CNN_ARC
        elif self.architecture=='mlstm-fcn':
            from models.mlstm_fcn import MLSTM_FCN as CNN_ARC
        elif self.architecture=='malstm-fcn':
            from models.malstm_fcn import  MALSTM_FCN as CNN_ARC

        from keras_tuner.tuners import BayesianOptimization
        tuner_fcn=True
        SEED = 1
        MAX_TRIALS = self.max_trials#search sayısı.
        EXECUTION_PER_TRIAL = 1 # aynı hiperparametre için kaç kez denesin.
        epochs_num = 2500 #epoch num
        models_dir = os.path.join(self.run_path, 'models')
        plots_path=os.path.join(self.run_path,'plots')
        ihtar_utils.make_dir(models_dir)
        ihtar_utils.make_dir(plots_path)
        self.X_shape=self.X_train.shape
        hypermodel=CNN_ARC(self.X_shape,self.lr,self.nb_classes) #LSTMhypermodel classından instance yarat.
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=self.patience) #aşırı öğrenmeyi engellemek için val_loss patience sayısı kadar gelişmezse durdur.
        if self.lr >= 8e-5:
            factor = 0.2
            plateau_patience = (self.patience / 2) - 2
        else:
            factor = 0.5
            plateau_patience = (self.patience / 2)
        ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                                                 patience=plateau_patience, verbose=1, min_lr=1e-8)
        df_batch_results = pd.DataFrame([])

        # lr_cosine=(tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-2,first_decay_steps=1000)) #SGD ,nesterov=True
        # lr_exp=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1,decay_steps=100000,decay_rate=0.96,staircase=True) #exponential decay
        #learning rate scheduler

        tuner=BayesianOptimization(hypermodel,objective=keras_tuner.Objective("val_loss",direction="min"),seed=SEED,max_trials=MAX_TRIALS,executions_per_trial=EXECUTION_PER_TRIAL, directory=os.path.normpath('C:/'),
                             project_name='/RS/İhtar_Model_Results_RS' + datetime.datetime.now().strftime("%Y%m%d_%H%M")) #BayessianOptimizasyon classından instance oluştur.
        tuner.search_space_summary()
        tuner.search(self.X_train,self.y_train,validation_data=(self.X_val,self.y_val),
                     batch_size=self.batch_size,epochs=epochs_num,callbacks=[stop_early,ReduceLROnPlateau],verbose=1) #aramaya başla.
        best_hps = tuner.get_best_hyperparameters(num_trials=self.number_of_models_to_save) #en iyi num_trials kadar parametreyi kaydet.

        config=ihtar_utils.config_creator(self.feature_columns,self.scaler,self.window_size,self.stride_size)

        #mixed batch
        batch_macro_f1_scores = []
        model_name_list=[]
        #neg_batch ivedik and macunkoy
        for i,trial in enumerate(best_hps): #kaydedilen en iyi parametreleri al ve modelleri eğit.
            num_run_model=f"{self.observation_name}_{self.batch_size}_{self.lr}_{self.start_time[-4:]}_{i}" #eşsiz model isimleri için dinamik yapı.
            model_path=os.path.join(models_dir,num_run_model)
            ihtar_utils.make_dir(model_path)
            summary_path = model_path + "/summary.txt" #model özetini txte kaydet.
            dump(self.scaler, open(os.path.join(model_path, 'scaler.bin'), 'wb')) #scaleri kaydet.
            print(f"scale_: {self.scaler.scale_}")
            print(f"mean_: {self.scaler.mean_}")
            model = tuner.hypermodel.build(trial) #modeli seçilen parametreler ile inşa et.
            tensorboard_dir = model_path+ "/tensorboard"
            tensorboard_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1) #profile_batch='500,520'
            # csv_logger=CSVLogger(model_path+'/log.csv',append=True,separator=",")
            history = model.fit(self.X_train,self.y_train,validation_data=(self.X_val,self.y_val),batch_size=self.batch_size ,epochs=epochs_num,
                                callbacks=[stop_early,ReduceLROnPlateau,tensorboard_callback]) #modeli eğit.

            save_fig(history,model_path,plots_path,i,self.observation_name) #loss ve acc grafiklerini çiz.
            model.save(model_path) #modeli kaydet.
            # tf.keras.utils.plot_model(model,to_file=model_path+"\model.png")

            hp_config={}
            hp_config['bs'] = self.batch_size
            hp_config['pos_ds_ratio']=self.pos_ds_ratio
            hp_config['neg_ds_ratio']=self.neg_ds_ratio
            # hp_config['ws'] = self.window_size
            # hp_config['stride'] = self.stride_size
            # hp_config['ds_ratio']=[pos_train_dts_ratio,neg_train_dts_ratio]
            hp_config['lr_rate'] = self.lr
            hp_config['ptnc']=self.patience
            hp_config.update(trial.values)

            with open(f'{model_path}\\config.ini', "w") as configfile:
                config.write(configfile)

            model_param = pd.DataFrame()
            model_param.loc[:,'Feature_List']=self.feature_columns
            model_param['Window_Size']=self.window_size
            model_param['Stride_Size']=self.stride_size
            model_param['scale_']=self.scaler.scale_ #C++ entegrasyon kodu için eğitimde kullanılan özniteliklerin standard sapmayı ve ortalamayı kaydet.
            model_param['mean_']=self.scaler.mean_
            model_param.to_csv(model_path + "\\config.csv")
            df_result=test.tensorflow_models(model,self.dataset_dict,i,hp_config)
            df_batch_results=pd.concat([df_batch_results,df_result])

        df_batch_results=df_batch_results.sort_values(by=["batch"],ascending=False)
        return df_batch_results
        # df_all.to_csv(self.run_path + "\\cross_val_test_results.csv") #cross val test sonuçlarını kaydet.


