import math
import matplotlib.pyplot as plt
import tensorflow as tf


class MODEL:
    def __init__(self,X_shape):
        self.X_shape=X_shape
        self.lstm_1=64
        self.lstm_2=32
        self.dense_1=32
        self.dense_2=1
        self.dense_1_act='tanh'
        self.dense_2_act='sigmoid'
        self.learning_rate=1e-2
        self.optimizer='Adam'
        # self.keras_tuner_log_dir="tuner/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # from keras_tuner import HyperModel
    def LSTM(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(None, self.X_shape[2]), return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        #lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
        model.compile(optimizer=Adam(learning_rate=2.5e-2),loss='binary_crossentropy',metrics=['accuracy','Precision','Recall','mae'])
        return model,self.lstm_1,self.lstm_2,self.dense_1,self.dense_2,self.dense_1_act,self.dense_2_act,self.learning_rate,self.optimizer


    def step_decay_lr(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(None, self.X_shape[2]), return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        initial_lr=0.1
        drop=0.5
        epochs_drop=10
        lr_scheduler=tf.keras.callbacks.LearningRateScheduler(lambda epoch: initial_lr*math.pow(drop,math.floor((1+epoch)/epochs_drop)))
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
        model.compile(optimizer=Adam(learning_rate=1e-2,),loss='binary_crossentropy',metrics=['accuracy','Precision','Recall'])
        return model,self.lstm_1,self.lstm_2,self.dense_1,self.dense_2,self.dense_1_act,self.dense_2_act,self.learning_rate,self.optimizer,lr_scheduler

    def exp_decay_lr(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(None, self.X_shape[2]), return_sequences=True))
        model.add(LSTM(32, return_sequences=True))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        initial_lr=0.1
        k=0.1
        lr_scheduler=tf.keras.callbacks.LearningRateScheduler(lambda epoch: initial_lr*math.exp(-k*epoch))
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
        model.compile(optimizer=Adam(learning_rate=1e-2),loss='binary_crossentropy',metrics=['accuracy','Precision','Recall','mae'])
        return model,self.lstm_1,self.lstm_2,self.dense_1,self.dense_2,self.dense_1_act,self.dense_2_act,self.learning_rate,self.optimizer,lr_scheduler

LR_START=1e-6
LR_MAX=6e-4
LR_MIN=1e-6
LR_RAMPUP_EPOCHS=0
LR_SUSTAIN_EPOCHS=0
EPOCHS=420
STEPS=[60,120,240]

def lrfn(epoch):
    if epoch<STEPS[0]:
        epoch2=epoch
        EPOCHS2=STEPS[0]
    elif epoch<STEPS[0]+STEPS[1]:
        epoch2=epoch-STEPS[0]
        EPOCHS2=STEPS[1]

    if epoch2 < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch2 + LR_START
    elif epoch2 < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch2 - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
      format(lr_y[0], max(lr_y), lr_y[-1]))
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
plt.xlabel('Epoch', size=14)
plt.ylabel('Learning Rate', size=14)
plt.show()