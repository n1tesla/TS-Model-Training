from tensorflow.keras.layers import Dense,\
    Conv1D,BatchNormalization,GlobalAveragePooling1D,Input,ReLU,Add
from tensorflow.keras.optimizers import Adam
import os

os.environ["KERAS_BACKEND"] = "theano"
from keras_tuner import HyperModel

class RESNET(HyperModel):
    def __init__(self,input_shape,lr,nb_classes):
        self.input_shape=input_shape[1:]
        self.nb_classes=nb_classes
        self.lr=lr

    def build(self,hp):
        #build conv_x
        input_layer=Input(shape=(self.input_shape))
        conv_x=BatchNormalization()(input_layer)
        conv_x=Conv1D(filters=hp.Choice(f"CNN_x_filters",values=[16,32,64,128]),kernel_size=hp.Choice(f"layerx_kernel_size"),values=[3,5,8],padding='same')(conv_x)
        conv_x=BatchNormalization()(conv_x)
        conv_x=ReLU()(conv_x)

        #build conv_y
        conv_y = Conv1D(filters=hp.Choice(f"CNN_y_filters", values=[16, 32, 64, 128]),
                        kernel_size=hp.Choice(f"layery_kernel_size"), values=[3, 5, 8], padding='same')(conv_x)
        conv_y=BatchNormalization()(conv_y)
        conv_y=ReLU()(conv_y)

        #build conv_z
        conv_z = Conv1D(filters=hp.Choice(f"CNN_z_filters", values=[16, 32, 64, 128]),
                        kernel_size=hp.Choice(f"layerz_kernel_size"), values=[3, 5, 8], padding='same')(conv_y)
        conv_z=BatchNormalization()(conv_z)


        is_expand_channels=not (self.input_shape[-1]==64*2)
        if is_expand_channels:
            shortcut_y=Conv1D(128,kernel_size=1,padding='same')(input_layer)
            shortcut_y=BatchNormalization()(shortcut_y)
        else:
            BatchNormalization()(input_layer)
        y=Add()([shortcut_y,conv_z])
        y=ReLU()(y)





