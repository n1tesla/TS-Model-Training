import numpy as np
import pandas as pd
import os
import save_plots

pd.options.mode.chained_assignment = None
class SPLITTER:
    def __init__(self,feature_columns,label_features,pos_dataset_ratio,neg_dataset_ratio,window_size,stride,step_size,run_path,csv_number):
        self.feature_columns = feature_columns
        self.label_features = label_features
        self.pos_dataset_ratio = pos_dataset_ratio
        self.neg_dataset_ratio=neg_dataset_ratio
        self.window_size, self.stride =window_size,stride
        self.step_size=step_size
        self.path=run_path
        self.csv_number=csv_number

    def show_csv(self):
        #eğitime girecek csv dosyaların ismini sözlüğe at.
        input_dir_list = list()
        for dirname, _, filenames in os.walk('../train'):
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
        test_pos = 0
        Dataset_dict=self.show_csv()

        #train, val ve test csv dosyalarını oluştur.
        df_train = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_val = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_test = pd.DataFrame(columns=self.label_features + self.feature_columns)
        neg_csv=pd.DataFrame()
        pos_csv=pd.DataFrame()

        #csv dosyalarını kendi içinde cross-val tekniği kullan.
        for key, dataset_name in enumerate(Dataset_dict.keys()):
            df = Dataset_dict[dataset_name]
            df['split_id'] = np.zeros(len(df.index)) #csv in içinde yer alan veri sayısı kadar split_id yi 0 olarak başlat.
            df_pos = df[df['label'] == 1]
            df_neg = df[df['label'] == 0]
            print(f"{dataset_name} NEG: {len(df_neg.index)}\t POS: {len(df_pos.index)}")
            if len(df_neg.index)!=0: #negatif cross-val split.
                neg_train = 0
                neg_val = 0
                neg_test = 0
                size_neg = len(df_neg.index)
                total_neg += size_neg
                UniqueIds = df_neg['id'].unique()
                for id in UniqueIds: #bir id nin tamamını train,val veya teste at. Veriseti oranına göre, Doldurma sırası, test, val, train.
                    split_id += 1
                    SplitDataIdx = df_neg['id'].isin([id])
                    SplitDataIdx = df_neg.loc[SplitDataIdx, :]
                    if len(SplitDataIdx.index) < self.window_size:
                        continue
                    SplitDataIdx['split_id'] = np.ones(len(SplitDataIdx.index)) * split_id

                    if  neg_test<self.dataset_ratio[2]*size_neg: #neg_test sayısı belirlenen orandan büyük olana kadar testi doldur.
                        neg_test+=len(SplitDataIdx.index)
                        df_test=pd.concat([df_test,SplitDataIdx])
                        continue
                    elif neg_val<self.dataset_ratio[1]*size_neg and neg_test>self.dataset_ratio[2]*size_neg: #neg val sayısı belirlenen orandan büyük olana kadar neg_val i doldur.
                        neg_val+=len(SplitDataIdx.index)
                        df_val=pd.concat([df_val,SplitDataIdx])
                        continue
                    #neg train sayısı belirlenen orandan büyük olana kadar neg_traini i doldur.
                    elif neg_train<self.dataset_ratio[0]*size_neg and neg_test>self.dataset_ratio[2]*size_neg and neg_val>self.dataset_ratio[1]*size_neg:
                        neg_train+=len(SplitDataIdx.index)
                        df_train=pd.concat([df_train,SplitDataIdx])
                    else:
                        neg_train+len(SplitDataIdx.index)
                        df_train=pd.concat([df_train,SplitDataIdx])

                data_neg = pd.DataFrame(data={'Dataset_Name':dataset_name,'neg':size_neg,'train_neg':neg_train,'val_neg':neg_val,'test_neg':neg_test},index=[key])
                neg_csv=pd.concat([neg_csv,data_neg])

            #pos cross validation.
            if len(df_pos.index)!=0:
                len_pos = len(df_pos.index)
                total_pos += len_pos
                train_step = round(len_pos * (self.dataset_ratio[0] / self.step_size))
                val_step = round(len_pos * (self.dataset_ratio[1] / self.step_size))
                test_step = round(len_pos * (self.dataset_ratio[2] / self.step_size))
                print(f"train_step: {train_step}\t val_step:{val_step}\t test_step:{test_step}")
                #train_step+val_step+test_step büyüklüğünde adım at.
                for i in range(0, len_pos, train_step + val_step + test_step):
                    if len_pos - i > self.window_size: #kalan veri window_sizedan büyük mü?
                        if len_pos - i >= train_step + val_step + test_step: #kalan veri train_step + val_step + test_step toplamından büyük mü?
                            train_pos += train_step
                            val_pos += val_step
                            test_pos += test_step
                            split_id += 1 #her bir split için farklı id belirle.
                            df_pos[i:i + train_step]['split_id'] = np.ones(train_step) * split_id #train veri sayısı kadar split_id ata.
                            split_id += 1
                            df_pos[i + train_step:i + train_step + val_step]['split_id'] = np.ones(val_step) #val veri sayısı kadar split_id ata.
                            split_id += 1
                            df_pos[i + train_step + val_step:i + train_step + val_step + test_step]['split_id'] = np.ones(test_step) * split_id #test veri sayısı kadar split_id ata.
                            df_train = pd.concat([df_train, df_pos[i:i + train_step]]) #belirlenen train verisini df_train e at
                            df_val = pd.concat([df_val, df_pos[i + train_step:i + train_step + val_step]]) #belirlenen val verisini df_train e at
                            df_test = pd.concat([df_test, df_pos[i + train_step + val_step:i + train_step + val_step + test_step]]) #belirlenen test verisini df_train e at
                            continue
                        elif len_pos - i >= train_step + val_step:    #kalan veri train_step + val_step toplamından büyük mü?
                            train_pos += train_step
                            val_pos += val_step
                            test_pos += len_pos - (i + train_step + val_step)
                            split_id += 1
                            df_pos[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                            split_id += 1
                            df_pos[i + train_step:i + train_step + val_step]['split_id'] = np.ones(val_step) * split_id
                            split_id += 1
                            df_train = pd.concat([df_train, df_pos[i:i + train_step]])
                            df_val = pd.concat([df_val, df_pos[i + train_step:i + train_step + val_step]])
                            if len_pos - (i + train_step + val_step) >= self.window_size:
                                df_pos[i + train_step + val_step:len_pos]['split_id'] = np.ones(
                                    len_pos - (i + train_step + val_step)) * split_id
                                df_test = pd.concat([df_test, df_pos[i + train_step + val_step:len_pos]])
                            continue
                        elif len_pos - i >= train_step:  #kalan veri train_step toplamından büyük mü?
                            train_pos += train_step
                            val_pos = len_pos - (i + train_step)
                            split_id += 1
                            df_pos[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                            split_id += 1
                            df_train = pd.concat([df_train, df_pos[i:i + train_step]])
                            if len_pos - i >= self.window_size:
                                df_pos[i + train_step:len_pos]['split_id'] = np.ones(len_pos - (i + train_step)) * split_id
                                df_val = pd.concat([df_val, df_pos[i + train_step:len_pos]])
                            continue
            if key>=1:
                #her bir csv dosyasından kaçı traine, val'a, teste gittiğini gösteren tablo.
                print(f"train_pos: {train_pos}\tval_pos: {val_pos}\ttest_pos: {test_pos}")
                pos_train=train_pos-pos_csv.loc[0:key]['train_pos'].sum()
                pos_val=val_pos-pos_csv.loc[0:key]['val_pos'].sum()
                pos_test=test_pos-pos_csv.loc[0:key]['test_pos'].sum()
                data_pos=pd.DataFrame(data={'pos':len(df_pos.index),'train_pos':pos_train,'val_pos':pos_val,'test_pos':pos_test,'pos_train_step':train_step,'pos_val_step':val_step,'pos_test_step':test_step},index=[key])
                pos_csv=pd.concat([pos_csv,data_pos])
            else:
                data_pos=pd.DataFrame(data={'pos':len(df_pos.index),'train_pos':train_pos,'val_pos':val_pos,'test_pos':test_pos,'pos_train_step':train_step,'pos_val_step':val_step,'pos_test_step':test_step},index=[key])
                pos_csv=pd.concat([pos_csv,data_pos])
        #df_config dosyasını oluştur.
        df_config=pd.concat([neg_csv,pos_csv],axis=1)
        df_config=df_config[['Dataset_Name','pos','train_pos','val_pos','test_pos','neg','train_neg','val_neg','test_neg','pos_train_step','pos_val_step','pos_test_step']]
        data_path=self.path+"\\data"
        try:
            os.mkdir(data_path)
        except Exception as e:
            print(e)

        train_neg = len(df_train[df_train['label'] == 0].index)
        val_neg = len(df_val[df_val['label'] == 0].index)
        test_neg = len(df_test[df_test['label'] == 0].index)

        print(f"Splitten önce toplam NEGATİF sayisi: {total_neg}")
        print(f"Splitten sonra NEGATİF dağılımı: neg_train: {train_neg} neg_val: {val_neg} neg_test {test_neg} neg_total: {train_neg + val_neg + test_neg}")

        total = pd.DataFrame({'Dataset_Name':'total','pos':total_pos,'train_pos':train_pos,'val_pos':val_pos,'test_pos':test_pos,'neg':total_neg,'train_neg':train_neg,'val_neg':val_neg,'test_neg':test_neg,
                              'pos_train_step':0,'pos_val_step':0,'pos_test_step':0},index=[key+1])
        df_config=pd.concat([df_config,total])
        df_config['dataset_ratio']=np.zeros(key+2)
        df_config.loc[:2,'dataset_ratio']=self.dataset_ratio
        df_config['step_size']=np.zeros(key+2)
        df_config.loc[:0,'step_size']=self.step_size
        df_config.to_csv(data_path+"\\df_config.csv")
        df_train.to_csv(data_path+"\\train.csv")
        df_val.to_csv(data_path+"\\val.csv")
        df_test.to_csv(data_path+"\\test.csv")

        #model parametrelerini datanın içinede kaydet.
        model_param = pd.DataFrame()
        model_param.loc[:, 'Feature_List'] = self.feature_columns
        model_param['Window_Size'] = self.window_size
        model_param['Stride_Size'] = self.stride
        model_param.to_csv(data_path + "\\config.csv")

        #veri dağılımı gösteren pasta grafiğini çiz.
        total = [train_pos + train_neg, val_pos + val_neg, test_pos + test_neg]
        neg_pos = [train_pos, train_neg, val_pos, val_neg, test_pos, test_neg]
        save_plots.distribution_pie(total,neg_pos,data_path)

        # TODO : 16 sampledan az değerleri pastaya ekleme
        return df_train,df_val,df_test










