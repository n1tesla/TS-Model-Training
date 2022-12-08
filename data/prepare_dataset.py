import numpy as np
import torch


class PREPARE:
    def __init__(self,feature_columns,label_features,window_size,stride,batch_size,shuffle_size,df_train,df_val,df_batch,neg_azimuth,pos_azimuth,synt_pos_0_45x,synt_pos_0_7x,synt_pos_1_2x,synt_pos_1_45x,synt_pos_1_7x,df_ivedik,df_macunkoy,architecture):

        self.feature_columns=feature_columns
        self.label_features = label_features
        self.window_size, self.stride =window_size,stride
        self.batch_size=batch_size
        self.shuffle_size=shuffle_size
        self.df_train=df_train
        self.df_val=df_val
        self.df_batch=df_batch
        self.neg_azimuth=neg_azimuth
        self.pos_azimuth=pos_azimuth
        self.synt_pos_0_45x = synt_pos_0_45x
        self.synt_pos_0_7x = synt_pos_0_7x
        self.synt_pos_1_2x = synt_pos_1_2x
        self.synt_pos_1_45x = synt_pos_1_45x
        self.synt_pos_1_7x = synt_pos_1_7x
        self.df_ivedik=df_ivedik
        self.df_macunkoy=df_macunkoy

    def create_lstm_dataset(self):
        X_train, y_train = self.overlap_TrackID(self.df_train,'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        X_val, y_val = self.overlap_TrackID(self.df_val, 'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        batch_test_X, batch_test_y = self.overlap_TrackID(self.df_batch,'id')  # id ye göre pencere kaydırma işlemini yap.
        X_shape = X_train.shape
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(self.shuffle_size).batch(
            self.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(self.shuffle_size).batch(
            self.batch_size)
        print(f"train_dataset: {train_dataset}")
        print(f"val_dataset: {val_dataset}")
        return train_dataset, val_dataset, X_shape, batch_test_X, batch_test_y

    def create_fcn_dataset(self):

        X_train, y_train = self.overlap_TrackID_FCN(self.df_train,'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        idx = np.random.permutation(len(X_train))
        X_train,y_train=X_train[idx],y_train[idx]

        X_val, y_val = self.overlap_TrackID_FCN(self.df_val, 'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        idx_val=np.random.permutation(len(X_val))
        X_val,y_val=X_val[idx_val],y_val[idx_val]
        batch_test_X, batch_test_y = self.overlap_TrackID_FCN(self.df_batch,'id')  # id ye göre pencere kaydırma işlemini yap.

        #synthetic azimuth change
        neg_az_X,neg_az_y=self.overlap_TrackID_FCN(self.neg_azimuth,'id')
        pos_az_X, pos_az_y = self.overlap_TrackID_FCN(self.pos_azimuth, 'id')
        # synthetic velocity change
        pos_0_45_X, pos_0_45_y = self.overlap_TrackID_FCN(self.synt_pos_0_45x, 'id')
        pos_0_7_X, pos_0_7_y = self.overlap_TrackID_FCN(self.synt_pos_0_7x, 'id')
        pos_1_2_X, pos_1_2_y = self.overlap_TrackID_FCN(self.synt_pos_1_2x, 'id')
        pos_1_45_X, pos_1_45_y = self.overlap_TrackID_FCN(self.synt_pos_1_45x, 'id')
        pos_1_7_X, pos_1_7_y = self.overlap_TrackID_FCN(self.synt_pos_1_7x, 'id')
        ivedik_X, ivedik_y = self.overlap_TrackID_FCN(self.df_ivedik, 'id')
        macukoy_X, macukoy_y = self.overlap_TrackID_FCN(self.df_macunkoy, 'id')
        X_shape = X_train.shape
        nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))
        print(f"X_train:{X_train.shape}\ty_train:{y_train.shape}\tX_val:{X_val.shape}\ty_val:{y_val.shape}\tnb_classes:{nb_classes}")

        return X_train,y_train,X_val,y_val,batch_test_X, batch_test_y,X_shape,nb_classes,neg_az_X,neg_az_y,pos_az_X, pos_az_y,pos_0_45_X, \
               pos_0_45_y,pos_0_7_X, pos_0_7_y,pos_1_2_X, pos_1_2_y ,pos_1_45_X, pos_1_45_y,pos_1_7_X, pos_1_7_y,ivedik_X, ivedik_y,macukoy_X, macukoy_y

    def overlap_TrackID_FCN(self,df,based_id):
        WindowedX, WindowedY = [], []
        UniqueSplitIds = df[based_id].unique()
        for unique in UniqueSplitIds:
            Split_Data = df[based_id].isin([unique])
            SplitDataX = df.loc[Split_Data, self.feature_columns].to_numpy()
            SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
            #print(f"SplitDataX: {SplitDataX.shape[0]},SplitDataY: {SplitDataY[0]},split:{unique}")
            if SplitDataX.shape[0] >= self.window_size:
                WindowedX.append(self.extract_window(SplitDataX, self.window_size,self.stride))
                WindowedY.append(self.extract_window(SplitDataY,  self.window_size,self.stride))
        WindowedX = np.concatenate(WindowedX, axis=0)
        WindowedY = np.concatenate(WindowedY, axis=0)
        CNN_y=np.array([i.mean() for i in WindowedY])

        return WindowedX, CNN_y

    def tsai_dataset(self):
        x_train, y_train = self.overlap_TrackID_FCN(self.df_train, 'split_id')
        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]

        x_val, y_val = self.overlap_TrackID_FCN(self.df_val,
                                                'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        idx_val = np.random.permutation(len(x_val))
        x_val, y_val = x_val[idx_val], y_val[idx_val]

        batch_test_X, batch_test_y = self.overlap_TrackID_FCN(self.df_batch,
                                                              'id')  # id ye göre pencere kaydırma işlemini yap.
        idx_test = np.random.permutation(len(batch_test_X))
        x_test, y_test = batch_test_X[idx_test], batch_test_y[idx_test]

        nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))
        return x_train,y_train,x_val,y_val,x_test,y_test,nb_classes

    def overlapTRACKID_W_Padding(self,df,based_id):
        WindowedX, WindowedY = [], []
        UniqueSplitIds = df[based_id].unique()
        for unique in UniqueSplitIds:
            Split_Data = df[based_id].isin([unique])
            SplitDataX = df.loc[Split_Data, self.feature_columns].to_numpy()
            SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
            print(SplitDataX.shape)

            if self.padding_threshold<=SplitDataX.shape[0]<self.window_size:
                example=SplitDataX
                example_2=SplitDataY
                pad = np.zeros((1, len(self.feature_columns)), dtype='float64')
                label=SplitDataY[0]
                pad_2=np.ones(1,dtype='float64')*label
                for _ in range(0,self.window_size-SplitDataX.shape[0]):
                    example=np.concatenate((example,pad),axis=0)
                    example_2=np.concatenate((example_2,pad_2))

                if example.shape[0]!=16 or example_2.shape[0]!=16:
                    raise Exception(f"Example shape is not equal to 16, {example.shape[0]}")

                WindowedX.append(np.expand_dims(example,0))
                WindowedY.append(np.expand_dims(example_2,0))

            elif SplitDataX.shape[0]>=self.window_size:
                WindowedX.append(self.extract_window(SplitDataX, self.window_size, self.stride))
                WindowedY.append(self.extract_window(SplitDataY, self.window_size, self.stride))

        WindowedX = np.concatenate(WindowedX, axis=0)
        WindowedY = np.concatenate(WindowedY, axis=0)
        CNN_y = np.array([i.mean() for i in WindowedY])
        return WindowedX, CNN_y

    def overlap_TrackID(self,df,based_id):
        WindowedX, WindowedY = [], []
        UniqueSplitIds = df[based_id].unique()
        for unique in UniqueSplitIds:
            Split_Data = df[based_id].isin([unique])
            SplitDataX = df.loc[Split_Data, self.feature_columns].to_numpy()
            SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
            #print(f"SplitDataX: {SplitDataX.shape[0]},SplitDataY: {SplitDataY[0]},split:{unique}")
            if SplitDataX.shape[0] >= self.window_size:
                WindowedX.append(self.extract_window(SplitDataX, self.window_size,self.stride))
                WindowedY.append(self.extract_window(SplitDataY,  self.window_size,self.stride))
        WindowedX = np.concatenate(WindowedX, axis=0)
        WindowedY = np.concatenate(WindowedY, axis=0)

        return WindowedX, WindowedY

    def extract_window(self,arr, size, stride):
        examples = []
        min_len = size - 1
        max_len = len(arr) - size
        for i in range(0, max_len + 1, stride):
            example = arr[i:size + i]
            examples.append(np.expand_dims(example, 0))
        return np.vstack(examples)

        # if self.pad_seq:
        #     X_train, y_train = self.overlapTRACKID_W_Padding(self.df_train,'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        #     idx = np.random.permutation(len(X_train))
        #     X_train = X_train[idx]
        #     y_train = y_train[idx]
        #     X_val, y_val = self.overlapTRACKID_W_Padding(self.df_val,'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        #     idx_val = np.random.permutation(len(X_val))
        #     X_val = X_val[idx_val]
        #     y_val = y_val[idx_val]
        # X_train=tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post')  # TODO dtype sorun çıkartabilir.
        # X_val=tf.keras.preprocessing.sequence.pad_sequences(X_val, padding='post')

    def create_torch_dataset(self,tensor_format=True):

        x_train,y_train=self.overlap_TrackID_FCN(self.df_train,'split_id')
        idx=np.random.permutation(len(x_train))
        x_train,y_train=x_train[idx],y_train[idx]

        x_val, y_val = self.overlap_TrackID_FCN(self.df_val, 'split_id')  # split id ye göre pencere kaydırma işlemini yap.
        idx_val=np.random.permutation(len(x_val))
        x_val,y_val=x_val[idx_val],y_val[idx_val]

        batch_test_X, batch_test_y = self.overlap_TrackID_FCN(self.df_batch,'id')  # id ye göre pencere kaydırma işlemini yap.
        idx_test=np.random.permutation(len(batch_test_X))
        x_test,y_test=batch_test_X[idx_test],batch_test_y[idx_test]

        nb_classes = len(np.unique(np.concatenate((y_train, y_val), axis=0)))

        ts=np.concatenate((x_train,x_val,x_test),axis=0)

        ts=np.transpose(ts,axes=(0,2,1))
        labels = np.concatenate((y_train, y_val, y_test), axis=0)
        print(f"X_train:{x_train.shape}\ty_train:{y_train.shape}\tX_val:{x_val.shape}\ty_val:{y_val.shape}\ty_test:{y_test.shape}\tnb_classes:{nb_classes}")

        train_shape=x_train.shape
        val_shape=x_val.shape
        test_shape=x_test.shape
        # TODO : bunlara +1 eklememiz gerekebilir.
        idx_train=range(train_shape[0])
        idx_val=range(train_shape[0],train_shape[0]+val_shape[0])
        idx_test=range(train_shape[0]+val_shape[0],train_shape[0]+val_shape[0]+test_shape[0])

        if tensor_format:
            ts=torch.FloatTensor(np.array(ts))
            labels=torch.LongTensor(labels)
            x_train=torch.LongTensor(x_train)
            y_train=torch.LongTensor(y_train)
            x_val=torch.LongTensor(x_val)
            y_val=torch.LongTensor(y_val)
            x_test=torch.LongTensor(x_test)
            y_test=torch.LongTensor(y_test)

            idx_train=torch.LongTensor(idx_train)
            idx_val=torch.LongTensor(idx_val)
            idx_test=torch.LongTensor(idx_test)
        return  ts,labels,idx_train,idx_val,idx_test,nb_classes




