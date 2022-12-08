from tensorflow.keras.layers import Dense,Conv1D,BatchNormalization,GlobalAveragePooling1D,\
    Input,ReLU,Reshape,Masking,Permute,multiply,concatenate,Activation,LSTM,Dropout,Layer,Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
from keras import backend as K

os.environ["KERAS_BACKEND"] = "theano"

from keras_tuner import HyperModel

class Transormer(HyperModel):
    def __init__(self, input_shape,lr,nb_classes):
        self.input_shape = input_shape[1:]
        self.nb_classes=nb_classes
        self.lr=lr

    def build(self,hp):
        input_layer=Input(self.input_shape)

        x=Masking()(input_layer)
        LSTM_layer=LSTM(units=hp.Choice(f'LSTM_{1}_units', values=[8,16,32,64,128,256], default=128), return_sequences=True)(x)
        attention_layer=attention()(LSTM_layer)
        x=Dropout(rate=hp.Float(f'Dropout_LSTM_{1}', min_value=0, max_value=0.9, default=0.1, step=0.05))(attention_layer)

        y=Permute((2,1))(input_layer)
        y=Conv1D(filters=hp.Choice(f'CNN_1_filters',values=[16,32,64,128]),kernel_size=hp.Choice(f"layer1_kernel_size",values=[8]),padding='same',kernel_initializer='he_uniform')(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)
        y=self.squeeze_excite_block(y)

        y=Conv1D(filters=hp.Choice(f'CNN_2_filters',values=[32,64,128,256]),kernel_size=hp.Choice(f"layer2_kernel_size",values=[5]),padding='same',kernel_initializer='he_uniform')(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)
        y=self.squeeze_excite_block(y)

        y=Conv1D(filters=hp.Choice(f'CNN_3_filters',values=[16,32,64,128]),kernel_size=hp.Choice(f"layer3_kernel_size",values=[3]),padding='same',kernel_initializer='he_uniform')(y)
        y=BatchNormalization()(y)
        y=Activation('relu')(y)

        y=GlobalAveragePooling1D()(y)

        x=concatenate([x,y])
        out=Dense(self.nb_classes,activation='softmax')(x)
        model=Model(input_layer,out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=self.lr),
                      metrics=['sparse_categorical_accuracy'])
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

    # def create_LSTM_with_attention(self,hidden_units,dense_units,):

class attention(Layer):
    def __init__(self,**kwargs):

        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal",trainable=True)
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros",trainable=True)

        super(attention,self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e=K.squeeze(e,axis=-1)
        #compute the weights
        alpha=K.softmax(e)
        #reshape to tensorflow format
        alpha=K.expand_dims(alpha,axis=-1)
        #compute the context vector
        context=x*alpha
        context=K.sum(context,axis=1)
        return context

