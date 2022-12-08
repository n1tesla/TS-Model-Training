from tensorflow.keras.layers import Dense,LSTM, Dropout ,TimeDistributed,RepeatVector,\
    Conv1D,BatchNormalization,GlobalAveragePooling1D,Input,Activation,ReLU
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
import os

os.environ["KERAS_BACKEND"] = "theano"

from keras_tuner import HyperModel

class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape,lr):
        self.input_shape = input_shape
        self.lr=lr
    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Choice(f'LSTM_{1}_units', values=[64,128], default=128),
                                   input_shape=(self.input_shape[1], self.input_shape[2]), return_sequences=True))
        model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{1}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        model.add(LSTM(units=hp.Choice(f'LSTM_{2}_units', values=[32,64,96,128]), return_sequences=True))
        model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{2}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        model.add(LSTM(units=hp.Choice(f'LSTM_{3}_units', values=[16,32,64]),return_sequences=True))
        model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{3}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        model.add(LSTM(units=hp.Choice(f'LSTM_{4}_units', values=[16,32,64]), return_sequences=True))
        model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{4}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        model.add(LSTM(units=hp.Choice(f'LSTM_{5}_units', values=[16,32,64]),return_sequences=True))
        model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{5}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        model.add(LSTM(units=hp.Choice(f'LSTM_{6}_units', values=[16,32,64]), return_sequences=True))
        model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{6}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        model.add(LSTM())
        # model.add(LSTM(units=hp.Choice(f'LSTM_{7}_cells', values=[16,32]),return_sequences=True))
        # model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{7}', min_value=0, max_value=0.5, default=0.1, step=0.05)))
        # model.add(LSTM(units=hp.Choice(f'LSTM_{8}_cells', values=[16,32]),return_sequences=True))
        # model.add(Dropout(rate=hp.Float(f'Dropout_LSTM_{8}', min_value=0, max_value=0.5, default=0.1, step=0.05)))

        act_func1 = hp.Choice(name='act_func', values=['tanh'], ordered=False) #sigmoid ve tanh arasından en çok tanh i seçiyor.
        model.add(Dense(units=hp.Choice(f"Dense_{1}_units", values=[32,64,96,128]), activation=act_func1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=RMSprop(self.lr), loss='binary_crossentropy',metrics=['accuracy', 'Precision', 'Recall'])
        return model