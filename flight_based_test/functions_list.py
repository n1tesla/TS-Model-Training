import numpy as np


def extract_window(array,size,stride):
    examples=[]
    min_len = size-1
    max_len=len(array)-size

    # if window_type=="windowed_test":
    #     for j in range(0,min_len,stride):
    #         example = array[0:j+1]
    #         examples.append(np.expand_dims(example, 0))

    for i in range(0,max_len+1,stride):
        example = array[i:size+i]
        examples.append(np.expand_dims(example, 0))

    return np.vstack(examples)

def overlap_TrackID(df,based_id,window_size,stride,feature_columns,shape):
    WindowedX, WindowedY,PosXY = [], [],[]
    UniqueSplitIds = df[based_id].unique()
    for unique in UniqueSplitIds:
        Split_Data = df[based_id].isin([unique])
        SplitDataPosXY=df.loc[Split_Data,['pos_x','pos_y','label']].to_numpy()
        SplitDataX = df.loc[Split_Data, feature_columns].to_numpy()
        SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
        #print(f"SplitDataX: {SplitDataX.shape[0]},SplitDataY: {SplitDataY[0]},split:{unique}")
        if SplitDataX.shape[0] >= window_size:
            WindowedX.append(extract_window(SplitDataX, window_size,stride))
            WindowedY.append(extract_window(SplitDataY,  window_size,stride))
            PosXY.append(extract_window(SplitDataPosXY,window_size,stride))
    WindowedX = np.concatenate(WindowedX, axis=0)
    WindowedY = np.concatenate(WindowedY, axis=0)
    PosXY=np.concatenate(PosXY,axis=0)
    CNN_y = np.array([i.mean() for i in WindowedY])
    if shape==16:
        WindowedX=WindowedX.reshape((WindowedX.shape[0],window_size,len(feature_columns)))
    else:
        WindowedX=WindowedX.reshape((WindowedX.shape[0],len(feature_columns),window_size))
    return WindowedX, CNN_y,PosXY


