import numpy as np
import pandas as pd
import os
import save_plots
from sklearn import preprocessing
import random
from ihtar_utils import config_creator
pd.options.mode.chained_assignment = None
class CROSS_VAL_SPLIT:
    def __init__(self,feature_columns:list=None,pos_train_dts_ratio:float=None,neg_train_dts_ratio:float=None,window_size:int=16,
                 stride:int=4,run_path=None,csv_number:int=None,padding_seq:int=None,padding_threshold:int=None,dataset_path:str=None):
        self.feature_columns = feature_columns
        self.label_features = ["label", "id", "TrackTime"]
        self.pos_train_dts_ratio = pos_train_dts_ratio
        self.neg_train_dts_ratio=neg_train_dts_ratio
        self.window_size, self.stride =window_size,stride
        self.step_size=2
        self.path=run_path
        self.csv_number=csv_number
        self.padding_seq=padding_seq
        self.padding_threshold=padding_threshold
        self.dataset_path=dataset_path

    def get_csv(self)->dict:
        #eğitime girecek csv dosyaların ismini sözlüğe at.
        input_dir_list = list()
        for dirname, _, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                # if dirname.split('\\')[-1].startswith("angular_"):
                if filename.startswith("extractedData_All_Train"):
                    dir_tupple = (dirname, filename)
                    input_dir_list.append(dir_tupple)
        Dataset_dict = {}
        for input_dir in input_dir_list:
            dirname, filename = input_dir
            foldername = dirname.split('/')[-1]
            df = pd.read_csv(os.path.join(dirname, filename))
            Dataset_dict.update({foldername + ".csv": df})
            if len(Dataset_dict)==self.csv_number:
                break

        print(Dataset_dict.keys())
        return Dataset_dict

    def split(self):
        split_id=1 #cross-validation yaparken veriye atanan yeni id. Bu id'ye göre pencere kaydırma işlemi yapılıyor.
        #veri dağılımını görmek için.
        total_neg = 0
        total_pos = 0
        train_pos = 0
        val_pos = 0
        # test_pos = 0
        Dataset_dict=self.get_csv()

        #train, val ve test csv dosyalarını oluştur.
        df_train = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_val = pd.DataFrame(columns=self.label_features + self.feature_columns)
        # df_test = pd.DataFrame(columns=self.label_features + self.feature_columns)
        neg_csv=pd.DataFrame()
        pos_csv=pd.DataFrame()

        #csv dosyalarını kendi içinde cross-val tekniği kullan.
        # keys=list(Dataset_dict.keys())
        # random.shuffle(keys)
        l=list(Dataset_dict.items())
        random.shuffle(l)
        Dataset_dict=dict(l)

        for key, dataset_name in enumerate(Dataset_dict.keys()):
            df = Dataset_dict[dataset_name]
            df['split_id'] = np.zeros(len(df.index)) #csv in içinde yer alan veri sayısı kadar split_id yi 0 olarak başlat.
            df_pos = df[df['label'] == 1]
            df_neg = df[df['label'] == 0]

            print(f"{dataset_name} NEG: {len(df_neg.index)}\t POS: {len(df_pos.index)}")
            if len(df_neg.index)!=0: #negatif cross-val split.
                neg_train = 0
                neg_val = 0
                # neg_test = 0
                size_neg = len(df_neg.index)
                total_neg += size_neg
                UniqueIds = df_neg['id'].unique()
                for id in UniqueIds: #bir id nin tamamını train,val veya teste at. Veriseti oranına göre, Doldurma sırası, test, val, train.
                    split_id += 1
                    SplitDataIdx = df_neg['id'].isin([id])
                    SplitDataIdx = df_neg.loc[SplitDataIdx, :]
                    if self.padding_seq:
                        if len(SplitDataIdx.index) < self.padding_threshold:
                            continue
                    else:
                        if len(SplitDataIdx.index) < self.window_size:
                            continue

                    SplitDataIdx['split_id'] = np.ones(len(SplitDataIdx.index)) * split_id

                    if neg_val<(1-self.neg_train_dts_ratio)*size_neg: #neg val sayısı belirlenen orandan büyük olana kadar neg_val i doldur.
                        neg_val+=len(SplitDataIdx.index)
                        df_val=pd.concat([df_val,SplitDataIdx])
                        continue
                    #neg train sayısı belirlenen orandan büyük olana kadar neg_traini i doldur.
                    elif neg_train<self.neg_train_dts_ratio*size_neg and neg_val>(1-self.neg_train_dts_ratio)*size_neg:
                        neg_train+=len(SplitDataIdx.index)
                        df_train=pd.concat([df_train,SplitDataIdx])
                    else:
                        neg_train+len(SplitDataIdx.index)
                        df_train=pd.concat([df_train,SplitDataIdx])

                data_neg = pd.DataFrame(data={'Dataset_Name':dataset_name,'neg':size_neg,'train_neg':neg_train,'val_neg':neg_val},index=[key])
                neg_csv=pd.concat([neg_csv,data_neg])
            len_pos = len(df_pos.index)
            train_step = round(len_pos * (self.pos_train_dts_ratio / self.step_size)) - 1
            val_step = round(len_pos * ((1 - self.pos_train_dts_ratio) / self.step_size)) - 1
            #pos cross validation.
            if len_pos!=0:
                total_pos += len_pos
                print(f"train_step: {train_step}\t val_step:{val_step}")
                #train_step+val_step+test_step büyüklüğünde adım at.
                for i in range(0, len_pos, train_step + val_step):
                    if len_pos - i >= train_step + val_step: #kalan veri train_step + val_step + test_step toplamından büyük mü?
                        train_pos += train_step
                        val_pos += val_step
                        split_id += 1 #her bir split için farklı id belirle.
                        df_pos[i:i + train_step]['split_id'] = np.ones(train_step) * split_id #train veri sayısı kadar split_id ata.
                        split_id += 1
                        df_pos[i + train_step:i + train_step + val_step]['split_id'] = np.ones(val_step) #val veri sayısı kadar split_id ata.
                        df_train = pd.concat([df_train, df_pos[i:i + train_step]]) #belirlenen train verisini df_train e at
                        df_val = pd.concat([df_val, df_pos[i + train_step:i + train_step + val_step]]) #belirlenen val verisini df_train e at
                        continue
                    elif len_pos - i >= train_step:    #kalan veri train_step + val_step toplamından büyük mü?
                        print(f"len_pos-i>train_step: {len_pos - i}>{train_step}")
                        train_pos += train_step
                        split_id += 1
                        df_pos[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                        df_train = pd.concat([df_train, df_pos[i:i + train_step]])
                        continue
                    elif len_pos-i>=val_step:
                        print(f"len_pos-i>val_step: {len_pos-i}>{val_step}")

                        val_pos += val_step
                        split_id += 1
                        df_pos[i:i + val_step]['split_id'] = np.ones(val_step) * split_id
                        df_val = pd.concat([df_val, df_pos[i:i + val_step]])
                        continue
                    elif len_pos-i>=self.padding_threshold:
                        print(f"len_pos-i>padding_threshold: {len_pos-i}>{self.padding_threshold}")

                        # if len_pos-i>=self.window_size:
                        #     split_id+=1
                        #     df_pos[i + train_step:len_pos]['split_id'] = np.ones(len_pos - (i + train_step)) * split_id
                        #     df_val = pd.concat([df_val, df_pos[i + train_step:len_pos]])

            if key>=1:
                #her bir csv dosyasından kaçı traine, val'a, teste gittiğini gösteren tablo.
                print(f"train_pos: {train_pos}\tval_pos: {val_pos}")
                pos_train=train_pos-pos_csv.loc[0:key]['train_pos'].sum()
                pos_val=val_pos-pos_csv.loc[0:key]['val_pos'].sum()
                # pos_test=test_pos-pos_csv.loc[0:key]['test_pos'].sum()

                data_pos=pd.DataFrame(data={'pos':len(df_pos.index),'train_pos':pos_train,'val_pos':pos_val,'pos_train_step':train_step,'pos_val_step':val_step},index=[key])
                pos_csv=pd.concat([pos_csv,data_pos])
            else:
                data_pos=pd.DataFrame(data={'pos':len(df_pos.index),'train_pos':train_pos,'val_pos':val_pos,'pos_train_step':train_step,'pos_val_step':val_step},index=[key])
                pos_csv=pd.concat([pos_csv,data_pos])
        #df_config dosyasını oluştur.
        df_config=pd.concat([neg_csv,pos_csv],axis=1)
        df_config=df_config[['Dataset_Name','pos','train_pos','val_pos','neg','train_neg','val_neg','pos_train_step','pos_val_step']]
        data_path=self.path+"\\data"
        try:
            os.mkdir(data_path)
        except Exception as e:
            print(e)

        train_neg = len(df_train[df_train['label'] == 0].index)
        val_neg = len(df_val[df_val['label'] == 0].index)

        print(f"Number of Total Pos Data before Cross-Split: {total_pos}\tAfter Cross-Split: {train_pos+val_pos}")
        print(f"Number of Total Neg Data before CrossSplit: {total_neg}\tAfter Cross-Split: neg_total: {train_neg + val_neg} neg_train: {train_neg} neg_val: {val_neg} ")

        total = pd.DataFrame({'Dataset_Name':'total','pos':total_pos,'train_pos':train_pos,'val_pos':val_pos,'neg':total_neg,'train_neg':train_neg,'val_neg':val_neg,
                              'pos_train_step':0,'pos_val_step':0},index=[key+1])
        df_config=pd.concat([df_config,total])
        df_config.loc['pos_train_dts_ratio']=self.pos_train_dts_ratio
        df_config.loc['neg_train_dts_ratio']=self.neg_train_dts_ratio

        df_config.to_csv(data_path+"\\df_config.csv")
        df_train.to_csv(data_path+"\\train.csv")
        df_val.to_csv(data_path+"\\val.csv")

        #model parametrelerini datanın içinede kaydet.
        #veri dağılımı gösteren pasta grafiğini çiz.
        total = [train_pos + train_neg, val_pos + val_neg]
        neg_pos = [train_pos, train_neg, val_pos, val_neg]
        save_plots.distribution_pie(total,neg_pos,data_path)

        return df_train,df_val

class PREPARATION:
    def __init__(self,feature_columns:list=None,window_size:int=16,stride:int=4,df_train=None,df_val=None,include_extra:bool=True,mode:int=None,framework=None):

        self.feature_columns=feature_columns
        self.window_size, self.stride =window_size,stride
        self.df_train=df_train
        self.df_val=df_val
        self.dataset_dict={}
        self.include_extra=include_extra
        self.mode=mode
        self.framework=framework


    def create_dataset(self):
        self.dataset_dict['df_train']=self.df_train
        self.dataset_dict['df_val']=self.df_val
        if self.include_extra:
            self.include_extra_test_dataset(mode=self.mode)
        scaler=self.Standard_scaler()
        for k,v in self.dataset_dict.items():
            if k in ['df_train','df_val']:
                based_id='split_id'
            else:
                based_id='id'
            self.dataset_dict[k]=self.SlidingWindow(v,based_id,shuffle=True)


        return self.dataset_dict,scaler

    def SlidingWindow(self,df,based_id:str,shuffle:bool=None)->list:
        """sliding window help docs
        :param shuffle: means you can shuffle the train data
        """
        WindowedX, WindowedY, PosXY= [], [], []
        UniqueSplitIds = df[based_id].unique()

        for unique in UniqueSplitIds:
            Split_Data = df[based_id].isin([unique])
            SplitDataX = df.loc[Split_Data, self.feature_columns].to_numpy()
            SplitDataY = df.loc[Split_Data, 'label'].to_numpy()
            SplitDataPosXY=df.loc[Split_Data,['pos_x','pos_y','label']].to_numpy()
            # print(f"SplitDataX: {SplitDataX.shape[0]},SplitDataY: {SplitDataY[0]},split:{unique}")
            if SplitDataX.shape[0] >= self.window_size:
                WindowedX.append(self.extract_window(SplitDataX, self.window_size, self.stride))
                WindowedY.append(self.extract_window(SplitDataY, self.window_size, self.stride))
                PosXY.append(self.extract_window(SplitDataPosXY,self.window_size,self.stride))
        WindowedX = np.concatenate(WindowedX, axis=0)
        WindowedY = np.concatenate(WindowedY, axis=0)
        PosXY=np.concatenate(PosXY,axis=0)
        CNN_y = np.array([i.mean() for i in WindowedY])
        if based_id=='split_id' and shuffle==True:
            idx = np.random.permutation(len(WindowedX))
            WindowedX, CNN_y = WindowedX[idx], CNN_y[idx]
        if self.framework:  #tensorflow shape
            WindowedX=WindowedX.reshape((WindowedX.shape[0],self.window_size,len(self.feature_columns)))
        else:     #pytorch shape
            WindowedX = WindowedX.reshape((WindowedX.shape[0], len(self.feature_columns), self.window_size))

        return [WindowedX, CNN_y, PosXY] #TODO CONCATENATE EDİP DÖNDÜR, dictte tutmanın daha hafif bir yöntemi var mı?

    def extract_window(self,arr, size, stride):
        examples = []
        min_len = size - 1
        max_len = len(arr) - size
        for i in range(0, max_len + 1, stride):
            example = arr[i:size + i]
            examples.append(np.expand_dims(example, 0))
        return np.vstack(examples)

    def include_extra_test_dataset(self,mode):
        #TODO bunları TRY except blokların içine al
        # TEST data
        cwd=os.getcwd()
        if self.mode:
            batch=pd.read_csv(cwd+"\\data\\batch_test\\extractedData_All_Test.csv")
            neg_azimuth = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Neg_OnlyAz\\extractedData_Neg_TestBatch.csv")
            pos_azimuth = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Pos_OnlyAz\\extractedData_Pos_TestBatch.csv")
            pos_0_45x = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Pos_v0_45x\\extractedData_Pos_TestBatch.csv")
            pos_0_7x = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Pos_v0_7x\\extractedData_Pos_TestBatch.csv")
            pos_1_2x = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Pos_v1_2x\\extractedData_Pos_TestBatch.csv")
            pos_1_45x = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Pos_v1_45x\\extractedData_Pos_TestBatch.csv")
            pos_1_7x = pd.read_csv(cwd + "\\data\\batch_test\\SyntTest_Pos_v1_7x\\extractedData_Pos_TestBatch.csv")
            ivedik = pd.read_csv(cwd + "\\data\\batch_test\\ivedik_neg\\extractedData_All_Test.csv")
            macunkoy = pd.read_csv(cwd + "\\data\\batch_test\\macunkoy\\extractedData_All_Test.csv")

            self.dataset_dict['batch']=batch
            self.dataset_dict['neg_azimuth']=neg_azimuth
            self.dataset_dict['pos_azimuth']=pos_azimuth
            self.dataset_dict['pos_0_45x']=pos_0_45x
            self.dataset_dict['pos_0_7x']=pos_0_7x
            self.dataset_dict['pos_1_2x']=pos_1_2x
            self.dataset_dict['pos_1_45x']=pos_1_45x
            self.dataset_dict['pos_1_7x']=pos_1_7x
            self.dataset_dict['ivedik']=ivedik
            self.dataset_dict['macunkoy']=macunkoy

        else:

            default_path=cwd+"\\data\\test_fusion_dataset"
            batch=pd.read_csv(default_path+"\\batch\\extractedData_All_Train.csv")
            zikzak0=pd.read_csv(default_path+"\\06_30_12_24_zikzak\\extractedData_All_Train.csv")
            duz_takip_edilemeyen=pd.read_csv(default_path+"\\06_30_12_39_duz_takip_edilemeyen\\extractedData_All_Train.csv")
            cember=pd.read_csv(cwd+"\\data\\test_fusion_dataset\\06_30_13_07_cember\\extractedData_All_Train.csv")
            duz_sola0=pd.read_csv(default_path+"\\06_30_13_17_duz_sola\\extractedData_All_Train.csv")
            zikzak1 = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\06_30_13_20_zikzak\\extractedData_All_Train.csv")
            keskin_zikzak=pd.read_csv(default_path+"\\06_30_13_36_keskin_zikzak\\extractedData_All_Train.csv")
            kalp1 = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\06_30_13_58_kalp\\extractedData_All_Train.csv")
            just_neg = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_11_22_just_neg\\extractedData_All_Train.csv")
            duz_isinlanma=pd.read_csv(default_path+"\\07_01_11_35_duz_isinlanma\\extractedData_All_Train.csv")
            neg = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_11_56_neg\\extractedData_All_Train.csv")
            random1 = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_12_39_random\\extractedData_All_Train.csv")
            S = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_12_54_S\\extractedData_All_Train.csv")
            duz1 = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_13_02_duz\\extractedData_All_Train.csv")
            duz_sola = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_13_27_duz_sola\\extractedData_All_Train.csv")
            negatif = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_13_34_negatif\\extractedData_All_Train.csv")
            duz2 = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_13_51_duz\\extractedData_All_Train.csv")
            kalp2 = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_14_22_kalp\\extractedData_All_Train.csv")
            yedigen = pd.read_csv(cwd + "\\data\\test_fusion_dataset\\07_01_14_35_7gen\\extractedData_All_Train.csv")


            self.dataset_dict['batch']=batch
            self.dataset_dict['zikzak0']=zikzak0
            self.dataset_dict['duz_takip_edilemeyen'] = duz_takip_edilemeyen
            self.dataset_dict['cember']=cember
            self.dataset_dict['duz_sola0'] = duz_sola0
            self.dataset_dict['zikzak1']=zikzak1
            self.dataset_dict['keskin_zikzak'] = keskin_zikzak
            self.dataset_dict['kalp1']=kalp1
            self.dataset_dict['just_neg']=just_neg
            self.dataset_dict['duz_isinlanma'] = duz_isinlanma
            self.dataset_dict['neg']=neg
            self.dataset_dict['random1']=random1
            self.dataset_dict['S']=S
            self.dataset_dict['duz1']=duz1
            self.dataset_dict['duz_sola']=duz_sola
            self.dataset_dict['negatif'] = negatif
            self.dataset_dict['duz2']=duz2
            self.dataset_dict['kalp2']=kalp2
            self.dataset_dict['yedigen']=yedigen

    def Standard_scaler(self): #also known as z-normalization.
        cwd = os.getcwd()
        scaler = preprocessing.StandardScaler()
        for k, v in self.dataset_dict.items():
            if k in ['df_train']:
                v.loc[:, self.feature_columns] = scaler.fit_transform(v.loc[:, self.feature_columns].values)
            else:
                v.loc[:, self.feature_columns] = scaler.transform(v.loc[:, self.feature_columns].values)
            self.dataset_dict[k] = v
        return scaler










