from tensorflow.keras.layers import Dense,Conv1D,BatchNormalization,GlobalAveragePooling1D,ReLU,Reshape,Masking,Permute,multiply,concatenate,Activation,LSTM,Dropout,Input
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam
import os

# os.environ["KERAS_BACKEND"] = "theano"

from keras_tuner import HyperModel

class MLSTM_FCN(HyperModel):
    def __init__(self, input_shape,lr,nb_classes):
        self.input_shape = input_shape[1:]
        self.nb_classes=nb_classes
        self.lr=lr

    def build(self,hp):
        input_layer=Input(self.input_shape)

        x=Masking()(input_layer)
        x=LSTM(units=hp.Choice(f'LSTM_{1}_units', values=[8]))(x)
        # x=LSTM(8)(x)
        x=Dropout(0.8)(x)

        y=Permute((2,1))(input_layer)
        y=Conv1D(filters=hp.Choice(f'CNN_1_filters',values=[16,32,64]),kernel_size=hp.Choice(f"layer1_kernel_size",values=[3,5,8]),padding='same',kernel_initializer='he_uniform')(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)
        y=self.squeeze_excite_block(y)


        y=Conv1D(filters=hp.Choice(f'CNN_2_filters',values=[32,64,128]),kernel_size=hp.Choice(f"layer2_kernel_size",values=[3,5]),padding='same',kernel_initializer='he_uniform')(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)
        y=self.squeeze_excite_block(y)

        y=Conv1D(filters=hp.Choice(f'CNN_3_filters',values=[16,32,64]),kernel_size=hp.Choice(f"layer3_kernel_size",values=[3]),padding='same',kernel_initializer='he_uniform')(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)

        y=GlobalAveragePooling1D()(y)

        x=concatenate([x,y])
        out=Dense(self.nb_classes,activation='softmax')(x)
        model=Model(input_layer,out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=self.lr),metrics=['sparse_categorical_accuracy'])
        return model


    def squeeze_excite_block(self,input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor

        Returns: a keras tensor
        '''
        filters = input.shape[-1]  # channel_axis = -1 for TF

        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se
