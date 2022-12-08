import numpy as np
import pandas as pd
import math
import random

class AUGMENTATION:
    def __init__(self,feature_columns,label_features,padding_threshold,window_size,stride,mask_features,mask_ratio):

        self.feature_columns=feature_columns
        self.label_features=label_features
        self.padding_threshold=padding_threshold
        self.window_size=window_size
        self.stride=stride
        self.mask_features=mask_features
        self.mask_ratio=mask_ratio
    def synthetic(self,df_train,df_val):
        df_train_synthetic = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_val_synthetic = pd.DataFrame(columns=self.label_features + self.feature_columns)
        for index,df in enumerate([df_train,df_val]):
            UniqueIds=df['split_id'].unique()
            for split_id in UniqueIds:
                SplitDataIdx=df['split_id'].isin([split_id])
                SplitDataIdx=df.loc[SplitDataIdx,:]
                SplitDataX = SplitDataIdx.loc[:,self.feature_columns]
                SplitDataY=SplitDataIdx.loc[:,self.label_features+['split_id']]



    def masking(self,df_train,df_val):
        df_train_masked = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_val_masked = pd.DataFrame(columns=self.label_features + self.feature_columns)
        for index,df in enumerate([df_train,df_val]):
            UniqueIds=df['split_id'].unique()
            for split_id in UniqueIds:
                SplitDataIdx=df['split_id'].isin([split_id])
                SplitDataIdx=df.loc[SplitDataIdx,:]
                sample = SplitDataIdx
                length_of_id=len(SplitDataIdx.index)

                for i in self.mask_features:
                    index_of_column = sample.columns.get_loc(i)
                    random_index_to_mask=random.sample(range(0,len(sample.index)),int(length_of_id*self.mask_ratio))
                    sample.iloc[random_index_to_mask,[index_of_column]]=np.zeros(len(random_index_to_mask))

                if index==0:
                    df_train_masked=pd.concat([df_train_masked,sample],ignore_index=True)
                else:
                    df_val_masked=pd.concat([df_val_masked,sample],ignore_index=True)
        print("hey")
        return df_train_masked,df_val_masked

    def moving_average_padding(self, df_train, df_val):
        df_train_padded = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_val_padded = pd.DataFrame(columns=self.label_features + self.feature_columns)
        for index, df in enumerate([df_train, df_val]):
            UniqueIds = df['split_id'].unique()
            for split_id in UniqueIds:
                SplitDataIdx = df['split_id'].isin([split_id])
                SplitDataIdx = df.loc[SplitDataIdx, :]
                SplitDataX = SplitDataIdx.loc[:, self.feature_columns]
                SplitDataY = SplitDataIdx.loc[:, self.label_features + ['split_id']]

                if len(SplitDataIdx.index) < self.padding_threshold:  # 12
                    continue
                elif self.padding_threshold <= SplitDataX.shape[0] < self.window_size:
                    len_df = len(SplitDataIdx.index)
                    sample = SplitDataIdx
                    pad_size = self.window_size - len_df
                    # model=ARMA(SplitDataX)
                    # model=model.fit()

                    print(sample['vel'])
                    sample= sample.rolling(window=SplitDataX.shape[0],min_periods=self.window_size - SplitDataX.shape[0],method='table').mean()
                    sample['velEMA']=sample['vel'].rolling(window=SplitDataX.shape[0],min_periods=self.window_size-SplitDataX.shape[0]).mean()
                    print(sample['vel'])
                    for _ in range(0, pad_size):
                        sample = pd.concat([sample, SplitDataIdx.iloc[-1:]], ignore_index=True)

                    if index == 0:
                        df_train_padded = pd.concat([df_train_padded, sample], ignore_index=True)
                    else:
                        df_val_padded = pd.concat([df_val_padded, sample], ignore_index=True)
                elif SplitDataX.shape[0] >= self.window_size:
                    sample = self.pad_last_window(SplitDataIdx)
                    if index == 0:
                        df_train_padded = pd.concat([df_train_padded, sample], ignore_index=True)
                    else:
                        df_val_padded = pd.concat([df_val_padded, sample], ignore_index=True)
        return df_train_padded, df_val_padded

    def padding(self,df_train,df_val):
        df_train_padded=pd.DataFrame(columns=self.label_features+self.feature_columns)
        df_val_padded=pd.DataFrame(columns=self.label_features+self.feature_columns)

        for index,df in enumerate([df_train,df_val]):
            UniqueIds=df['split_id'].unique()
            for split_id in UniqueIds:
                SplitDataIdx=df['split_id'].isin([split_id])
                SplitDataIdx=df.loc[SplitDataIdx,:]
                SplitDataX = SplitDataIdx.loc[:,self.feature_columns]
                SplitDataY=SplitDataIdx.loc[:,self.label_features+['split_id']]

                if len(SplitDataIdx.index)<self.padding_threshold: #12
                    continue
                elif self.padding_threshold<=SplitDataX.shape[0]<self.window_size:
                    len_df = len(SplitDataIdx.index)
                    sample = SplitDataIdx
                    pad_size=self.window_size-len_df
                    for _ in range(0, pad_size):
                        sample = pd.concat([sample, SplitDataIdx.iloc[-1:]], ignore_index=True)

                    if index == 0:
                        df_train_padded = pd.concat([df_train_padded, sample], ignore_index=True)
                    else:
                        df_val_padded = pd.concat([df_val_padded, sample], ignore_index=True)
                elif SplitDataX.shape[0]>=self.window_size:
                    sample=self.pad_last_window(SplitDataIdx)
                    if index==0:
                        df_train_padded=pd.concat([df_train_padded,sample],ignore_index=True)
                    else:
                        df_val_padded=pd.concat([df_val_padded,sample],ignore_index=True)
        return df_train_padded,df_val_padded
    def pad_last_window(self,SplitDataIdx):
        sample=SplitDataIdx
        # sample_2=SplitDataIdx.iloc[-1:]
        # SplitDataIdx.iloc[-1:]['elevation'],SplitDataIdx.iloc[-1:]['vel'],SplitDataIdx.iloc[-1:]['acc'],SplitDataIdx.iloc[-1:]['range'],\
        # SplitDataIdx.iloc[-1:]['angularVel'],SplitDataIdx.iloc[-1:]['radialVel'],SplitDataIdx.iloc[-1:]['rcs']=pd.DataFrame([(0,0,0,0,0,0,0)])
        len_df=len(sample.index)
        step_size=math.floor((len_df-self.window_size)/self.stride)
        pad_size=self.stride-(len_df-(step_size*self.stride+self.window_size))
        # unpaded_df=df.loc[(step_size+1)*stride:,:]
        if pad_size<self.stride or pad_size!=0:
            # SplitDataIdx[self.feature_columns]=pd.DataFrame([(0,0,0,0,0,0,0)])
            # df_padding_features = pd.DataFrame(np.zeros((1,len(self.feature_columns+self.label_features+['split_id']))), columns=self.label_features+self.feature_columns+['split_id'])
            # padded_row = pd.concat([sample_2,df_padding_features],axis=1)

            for _ in range(0,pad_size):
                sample = pd.concat([sample, SplitDataIdx.iloc[-1:]],ignore_index=True)
        return sample

    def shuffling(self,df_train,df_val):
        df_train_shuffled=pd.DataFrame(columns=self.label_features+self.feature_columns)
        df_val_shuffled=pd.DataFrame(columns=self.label_features+self.feature_columns)
        # new_index = [1,0,2,3,4,5,6,7,8,9,10,11,12,13,15,14]

        for index,df in enumerate([df_train,df_val]):
            UniqueIds=df['split_id'].unique()
            for split_id in UniqueIds:
                SplitDataIdx=df['split_id'].isin([split_id])
                SplitDataIdx=df.loc[SplitDataIdx,:]
                # SplitDataX = SplitDataIdx.loc[:,self.feature_columns]
                # SplitDataY=SplitDataIdx.loc[:,self.label_features+['split_id']]
                sample=SplitDataIdx
                max_len=len(sample.index)

                if max_len>=self.window_size:

                    for i in range(0,max_len,self.window_size):
                        if max_len - i < self.window_size:
                                window=sample.iloc[i:]
                        else:
                            window = sample.iloc[i:self.window_size + i]
                            if random.random()<0.2:
                                new_index = []
                                new_index.append(window.index[1])
                                new_index.append(window.index[0])
                                for j in range(2,14):
                                    new_index.append(window.index[j])
                                new_index.append(window.index[15])
                                new_index.append(window.index[14])
                                window=window.reindex(new_index)

                        if index==0:
                            df_train_shuffled=pd.concat([df_train_shuffled,window],ignore_index=True)
                        else:
                            df_val_shuffled=pd.concat([df_val_shuffled,window],ignore_index=True)

                else:
                    window=sample
                    if index == 0:
                        df_train_shuffled = pd.concat([df_train_shuffled, window], ignore_index=True)
                    else:
                        df_val_shuffled = pd.concat([df_val_shuffled, window], ignore_index=True)


        return df_train_shuffled,df_val_shuffled



    def make_noise(self):
        pass
    def mixup(self):
        pass