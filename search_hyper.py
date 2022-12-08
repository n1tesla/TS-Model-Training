import optuna
from optuna.integration import FastAIPruningCallback
from tsai_ihtar.all import *
from imports import *
from optuna.integration import FastAIPruningCallback

class TUNER:
    def __init__(self,X,y,splits,epoch,batch_size,patience,arch_name):
        self.X=X
        self.y=y
        self.splits=splits
        # self.lr=lr
        self.epoch=epoch
        self.batch_size=batch_size
        self.patience=patience
        self.arch_name=arch_name

    def inception_plus_objective(self,trial: optuna.Trial):
        # Define search space here. More info here https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
        nf = trial.suggest_categorical('num_filters',
                                       [32, 64, 96,128])  # search through all categorical values in the provided list
        depth = trial.suggest_int('depth', 3, 9,
                                  step=3)  # search through all integer values between 3 and 9 with 3 increment steps
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5,
                                           step=.1)  # search through all float values between 0.0 and 0.5 with 0.1 increment steps
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        bottleneck=True
        residual=True
        #epoch=trial.suggest_int('epoch', 30,50)
        batch_tfms = TSStandardize(by_sample=True)   #BATCH TFMS KULLANMA. BİZ DAHA ÖNCESİNDE SCALER KULLANIYORUZ

        learn = TSClassifier(self.X, self.y, splits=self.splits, bs=self.batch_size,
                             arch=self.arch_name, arch_config={'nf': nf, 'fc_dropout': dropout_rate, 'depth': depth,'bottleneck':bottleneck},
                             metrics=accuracy, cbs=FastAIPruningCallback(trial))

        # with ContextManagers([learn.no_logging()]):  # [Optional] this prevents fastai from printing anything during training
        learn.fit_one_cycle(self.epoch, lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=self.patience),
                        ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])

        # Return the objective value
        return learn.recorder.values[-1][1]  # return the validation loss value of the last epoch

    def inception_objective(self,trial: optuna.Trial):
        # Define search space here. More info here https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
        nf = trial.suggest_categorical('num_filters',
                        [8,16,32,64,128,256])  # search through all categorical values in the provided list. default=32
        depth = trial.suggest_int('depth', 3, 9,
                                  step=3)  # search through all integer values between 3 and 9 with 3 increment steps #default=6
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-2,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        bottleneck=trial.suggest_categorical('bottleneck',[0])
        residual=trial.suggest_categorical('residual',[0])
        #epoch=trial.suggest_int('epoch', 30,50)
        batch_tfms = TSStandardize(by_sample=True)   #BATCH TFMS KULLANMA. BİZ DAHA ÖNCESİNDE SCALER KULLANIYORUZ

        learn = TSClassifier(self.X, self.y, splits=self.splits, bs=self.batch_size,
                             arch=self.arch_name, arch_config={'nf': nf, 'depth': depth,'bottleneck':bottleneck,
                                                               'residual':residual},
                             metrics=accuracy, cbs=FastAIPruningCallback(trial))

        # with ContextManagers([learn.no_logging()]):  # [Optional] this prevents fastai from printing anything during training
        learn.fit_one_cycle(self.epoch, lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=self.patience),
                        ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])

        # Return the objective value
        return learn.recorder.values[-1][1]  # return the validation loss value of the last epoch

    def fcn_objective(self,trial: optuna.Trial):
        l1=trial.suggest_categorical('filter_1',[32,64,128,256])
        l2=trial.suggest_categorical('filter_2',[32,64,128,256])
        l3=trial.suggest_categorical('filter_3',[32,64,128,256])
        ks_1=trial.suggest_categorical('ks_1',[7,5,3])
        ks_2=trial.suggest_categorical('ks_2',[7,5,3])
        ks_3=trial.suggest_categorical('ks_3',[5,3])
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        learn=TSClassifier(self.X,self.y,splits=self.splits,bs=self.batch_size,arch=self.arch_name,
                           arch_config={'layers':[l1,l2,l3],'kss':[ks_1,ks_2,ks_3]},metrics=accuracy,
                           cbs=FastAIPruningCallback(trial))
        learn.fit_one_cycle(self.epoch,lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=self.patience),
                        ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])

        return learn.recorder.values[-1][1]

    def fcnPlus_objective(self,trial:optuna.Trial):
        l1=trial.suggest_categorical('filter_1',[32,64,128,256])
        l2=trial.suggest_categorical('filter_2',[32,64,128,256])
        l3=trial.suggest_categorical('filter_3',[32,64,128,256])
        ks_1=trial.suggest_categorical('ks_1',[7,5,3])
        ks_2=trial.suggest_categorical('ks_2',[7,5,3])
        ks_3=trial.suggest_categorical('ks_3',[5,3])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5,
                                           step=.1)  # search through all float values between 0.0 and 0.5 with 0.1 increment steps

        learn=TSClassifier(self.X,self.y,splits=self.splits,bs=self.batch_size,arch=self.arch_name,
                           arch_config={'layers':[l1,l2,l3],'kss':[ks_1,ks_2,ks_3],'fc_dropout':dropout_rate},metrics=accuracy,
                           cbs=FastAIPruningCallback(trial))
        learn.fit_one_cycle(self.epoch,lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=self.patience),
                        ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])

        # parameters={'l1':l1,'l2':l2,'l3':l3,'ks_1':ks_1,'ks_2':ks_2,'ks_3':ks_3}
        return learn.recorder.values[-1][1]

    def ResNet_objective(self,trial:optuna.Trial):
        nf=trial.suggest_categorical('nf',[32,64,96,128,256])
        ks_1 = trial.suggest_categorical('ks_1', [7, 5, 3])
        ks_2 = trial.suggest_categorical('ks_2', [7, 5, 3])
        ks_3 = trial.suggest_categorical('ks_3', [5, 3])

        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        learn=TSClassifier(self.X,self.y,splits=self.splits,bs=self.batch_size,arch=self.arch_name,
                           arch_config={'nf':nf,'kss':[ks_1,ks_2,ks_3]},metrics=accuracy,
                           cbs=FastAIPruningCallback(trial))
        learn.fit_one_cycle(self.epoch,lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=self.patience),
                        ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])

        return learn.recorder.values[-1][1]

    def ResNetPlus_objective(self,trial:optuna.Trial):
        nf=trial.suggest_categorical('nf',[32,64,96,128,256])
        ks_1 = trial.suggest_categorical('ks_1', [7, 5, 3])
        ks_2 = trial.suggest_categorical('ks_2', [5, 3])
        ks_3 = trial.suggest_categorical('ks_3', [5, 3])
        dropout_rate=trial.suggest_float("dropout_rate",0.0,0.5,step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        learn=TSClassifier(self.X,self.y,splits=self.splits,bs=self.batch_size,arch=self.arch_name,
                           arch_config={'nf':nf,'kss':[ks_1,ks_2,ks_3],'fc_dropout':dropout_rate},metrics=accuracy,
                           cbs=FastAIPruningCallback(trial))
        learn.fit_one_cycle(self.epoch,lr_max=learning_rate,cbs=[EarlyStoppingCallback(monitor='valid_loss',patience=self.patience),
                        ReduceLROnPlateau(monitor='valid_loss',factor=10,patience=2)])
        return learn.recorder.values[-1][1]

    def rnn_fcn_objective(self,trial:optuna.Trial): #MLSTM-FCN, MGRU-FCN, MRNN-FCN
        l1 = trial.suggest_categorical('filter_1', [32, 64, 128, 256])
        l2 = trial.suggest_categorical('filter_2', [32, 64, 128, 256])
        l3 = trial.suggest_categorical('filter_3', [32, 64, 128, 256])
        ks_1 = trial.suggest_categorical('ks_1', [7, 5, 3])
        ks_2 = trial.suggest_categorical('ks_2', [7, 5, 3])
        ks_3 = trial.suggest_categorical('ks_3', [5, 3])
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5,step=.1)  # search through all float values between 0.0 and 0.5 with 0.1 increment steps

        learn = TSClassifier(self.X, self.y, splits=self.splits, bs=self.batch_size, arch=self.arch_name,
                             arch_config={'conv_layers': [l1, l2, l3], 'kss': [ks_1, ks_2, ks_3],
                                          'fc_dropout': dropout_rate}, metrics=accuracy,
                             cbs=FastAIPruningCallback(trial))
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        learn.fit_one_cycle(self.epoch, lr_max=learning_rate,
                            cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=self.patience),
                                 ReduceLROnPlateau(monitor='valid_loss', factor=10, patience=2)])
        return learn.recorder.values[-1][1]

    def os_cnn_objective(self,trial:optuna.Trial):
        l1=trial.suggest_categorical('layer1',[1024])
        l2=trial.suggest_categorical('layer2',[229376])
        layers=[l1,l2]
        learn = TSClassifier(self.X, self.y, splits=self.splits, bs=self.batch_size, arch=self.arch_name,
                             metrics=accuracy,arch_config={'layers':layers},
                             cbs=FastAIPruningCallback(trial))
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2,
                                            log=True)  # search through all float values between 0.0 and 0.5 in log increment steps
        learn.fit_one_cycle(self.epoch, lr_max=learning_rate,
                            cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=self.patience),
                                 ReduceLROnPlateau(monitor='valid_loss', factor=10, patience=2)])
        return learn.recorder.values[-1][1]




    def get_objective(self):
        if self.arch_name=='InceptionTimePlus':
            return self.inception_plus_objective
        elif self.arch_name=='InceptionTime':
            return self.inception_objective
        elif self.arch_name=='FCN':
            return self.fcn_objective
        elif self.arch_name=='FCNPlus':
            return self.fcnPlus_objective
        elif self.arch_name in ['MLSTM_FCN','MRNN_FCN','MGRU_FCN']:
            return self.rnn_fcn_objective
        elif self.arch_name=='OmniScaleCNN':
            return self.os_cnn_objective




    def get_arch_config(self,trial_params):
        #1D CNNs
        if self.arch_name == 'InceptionTimePlus':
            arch_config={'nf':trial_params['num_filters'],'fc_dropout':trial_params['fc_dropout'],'depth':trial_params['depth']}
            return arch_config
        elif self.arch_name=='InceptionTime':
            arch_config={'nf':trial_params['num_filters'],'depth':trial_params['depth'],
                         'bottleneck':trial_params['bottleneck'],'residual':trial_params['residual']}
            return arch_config
        elif self.arch_name == 'FCN':
            arch_config={'layers':[trial_params['filter_1'],trial_params['filter_2'],trial_params['filter_3']],
                         'kss':[trial_params['ks_1'],trial_params['ks_2'],trial_params['ks_3']]}
            return arch_config
        elif self.arch_name == 'FCNPlus':
            arch_config={'layers':[trial_params['filter_1'],trial_params['filter_2'],trial_params['filter_3']],
                         'kss':[trial_params['ks_1'],trial_params['ks_2'],trial_params['ks_3']],'fc_dropout':trial_params['dropout_rate']}
            return arch_config
        elif self.arch_name=='ResNetPlus':
            arch_config={'nf':trial_params['nf'],'kss':[trial_params['ks_1'],trial_params['ks_2'],
                        trial_params['ks_3']],'fc_dropout':trial_params['dropout_rate']}
            return arch_config
        #RNN-CNNS
        elif self.arch_name in ['MLSTM_FCN','MRNN_FCN','MGRU_FCN']:
            arch_config={'conv_layers':[trial_params['filter_1'],trial_params['filter_2'],trial_params['filter_3']],
                         'kss':[trial_params['ks_1'],trial_params['ks_2'],trial_params['ks_3']],'fc_dropout':trial_params['dropout_rate']}
            return arch_config
        elif self.arch_name == 'OmniScaleCNN':
            arch_config={'layers':trial_params['layers']}





