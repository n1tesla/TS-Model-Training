from tensorflow.keras.layers import Dense,\
    Conv1D,BatchNormalization,GlobalAveragePooling1D,Input,ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from custom_metrics import f1
# os.environ["KERAS_BACKEND"] = "theano"

from keras_tuner import HyperModel

class FCN(HyperModel):
    def __init__(self, input_shape,lr,nb_classes):
        self.input_shape = input_shape[1:]
        self.nb_classes=nb_classes
        self.lr=lr

    def build(self,hp):
        input_layer=Input(self.input_shape)

        conv1=Conv1D(filters=hp.Choice(
            f'CNN_1_filters',values=[16,32,64,128]),kernel_size=hp.Choice(f"layer1_kernel_size",values=[3,5,8]),padding='same')(input_layer)
        conv1=BatchNormalization()(conv1)
        conv1=ReLU()(conv1)

        conv2=Conv1D(filters=hp.Choice(
            f'CNN_2_filters',values=[32,64,128,256]),kernel_size=hp.Choice(f"layer2_kernel_size",values=[3,5]),padding='same')(conv1)
        conv2=BatchNormalization()(conv2)
        conv2=ReLU()(conv2)

        conv3=Conv1D(filters=hp.Choice(
            f'CNN_3_filters',values=[16,32,64,128]),kernel_size=hp.Choice(f"layer3_kernel_size",values=[3]),padding='same')(conv2)
        conv3=BatchNormalization()(conv3)
        conv3=ReLU()(conv3)

        gap_layer=GlobalAveragePooling1D()(conv3)

        output_layer=Dense(self.nb_classes,activation='softmax')(gap_layer)

        model=Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=self.lr),metrics=['sparse_categorical_accuracy'])
        return model

class stable_fcn:
    def __init__(self, input_shape, lr, nb_classes):
        self.input_shape = input_shape[1:]
        self.nb_classes = nb_classes
        self.lr = lr
    def build(self):
        input_layer=Input(self.input_shape)
        conv1=Conv1D(filters=128,kernel_size=8,padding='same')(input_layer)
        conv1=BatchNormalization()(conv1)
        cov1=ReLU()(conv1)

        conv2=Conv1D(filters=256,kernel_size=5,padding='same')(conv1)
        conv2=BatchNormalization()(conv2)
        conv2=ReLU()(conv2)

        conv3 = Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)

        gap_layer=GlobalAveragePooling1D()(conv3)
        output_layer=Dense(self.nb_classes,activation='softmax')(gap_layer)
        model=Model(inputs=input_layer,outputs=output_layer)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=self.lr),
                      metrics=[])
        return model