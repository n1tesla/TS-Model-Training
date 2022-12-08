import numpy as np
import pandas as pd
import os
import save_plots
pd.options.mode.chained_assignment = None
class SPLITTER:
    def __init__(self,feature_columns,label_features,train_ratio_pos,train_val_ratio_neg,window_size,stride,run_path,csv_number):
        self.feature_columns = feature_columns
        self.label_features = label_features

        self.train_ratio_pos=train_ratio_pos
        self.train_val_ratio_neg=train_val_ratio_neg
        self.window_size, self.stride =window_size,stride

        self.path=run_path
        self.csv_number=csv_number

    def show_csv(self):
        #eğitime girecek csv dosyaların ismini sözlüğe at.
        input_dir_list = list()
        for dirname, _, filenames in os.walk('train'):
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
        Dataset_dict=self.show_csv()

        #train, val ve test csv dosyalarını oluştur.
        df_train = pd.DataFrame(columns=self.label_features + self.feature_columns)
        df_val = pd.DataFrame(columns=self.label_features + self.feature_columns)
        # df_test = pd.DataFrame(columns=self.label_features + self.feature_columns)
        neg_csv=pd.DataFrame()
        pos_csv=pd.DataFrame()
        for key, dataset_name in enumerate(Dataset_dict.keys()):
            df = Dataset_dict[dataset_name]
            df['split_id']=np.zeros(len(df.index))
            df_pos = df[df['label'] == 1].loc[:, self.feature_columns + self.label_features]
            df_neg = df[df['label'] == 0].loc[:, self.feature_columns + self.label_features]
            print(f"{dataset_name} NEG: {len(df_neg.index)}\t POS: {len(df_pos.index)}")

            if len(df_neg.index) != 0:
                train_ratio = self.train_val_ratio_neg[0]
                val_ratio = 1-train_ratio
                if train_ratio>1:
                    raise Exception(f"train ratio cannot be bigger than 1, {train_ratio}")

                # test_ratio = 0.05
                neg_train = 0
                neg_val = 0
                # neg_test = 0
                size_neg = len(df_neg.index)
                total_neg += size_neg
                UniqueIds = df_neg['id'].unique()
                for id in UniqueIds:
                    split_id += 1
                    SplitDataIdx = df_neg['id'].isin([id])
                    SplitDataIdx = df_neg.loc[SplitDataIdx, :]
                    if len(SplitDataIdx.index) < self.window_size:
                        continue
                    SplitDataIdx['split_id'] = np.ones(len(SplitDataIdx.index)) * split_id
                    # print(f"{len(SplitDataIdx.index)}")
                    # if neg_test < test_ratio * size_neg:
                    #     neg_test += len(SplitDataIdx.index)
                    #     df_test = pd.concat([df_test, SplitDataIdx])
                    #     continue
                    if neg_val < val_ratio * size_neg:
                        neg_val += len(SplitDataIdx.index)
                        df_val = pd.concat([df_val, SplitDataIdx])
                        continue
                    elif neg_train < train_ratio * size_neg and neg_val > val_ratio * size_neg:
                        neg_train += len(SplitDataIdx.index)
                        df_train = pd.concat([df_train, SplitDataIdx])
                    else:
                        neg_train + len(SplitDataIdx.index)
                        df_train = pd.concat([df_train, SplitDataIdx])

                data_neg = pd.DataFrame(data={'Dataset_Name': dataset_name, 'neg': size_neg, 'train_neg': neg_train, 'val_neg': neg_val}, index=[key])
                neg_csv = pd.concat([neg_csv, data_neg])

            if len(df_pos.index) != 0:
                if self.window_size == 16:
                    # positive data split
                    # df_pos = df_pos.sort_values(by=['SystemTime'])
                    UniqueIds = df_pos['id'].unique()
                    for index, id in enumerate(UniqueIds):
                        train_ratio = self.train_ratio_pos[0]
                        val_ratio = 1- train_ratio

                        SplitDataIdx = df_pos['id'].isin([id])
                        SplitDataIdx = df_pos.loc[SplitDataIdx, :]
                        len_split_data = len(SplitDataIdx.index)
                        if len_split_data < self.window_size:
                            continue
                        SplitDataIdx['split_id'] = np.zeros(len_split_data)
                        if len_split_data >= self.window_size * 10:  # train,val test cross val yapmak için id nin en az 80 büyüklüğünde olması gerek
                            for step in range(4, 0,-1):  # dinamik step_size, id nin uzunluğuna göre step size belirliyorz.
                                if len_split_data > self.window_size * 10 * step:
                                    step_size = step
                                    break
                            train_step = round((len_split_data *train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)

                            if train_step < self.window_size or val_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step}")

                            print(f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}")
                            for i in range(0, len_split_data, train_step + val_step):
                                if len_split_data - i > self.window_size:
                                    if len_split_data - i >= train_step + val_step:
                                        train_pos += train_step
                                        val_pos += val_step

                                        split_id += 1
                                        SplitDataIdx[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                                        split_id += 1
                                        SplitDataIdx[i + train_step:i + train_step + val_step]['split_id'] = np.ones(val_step) * split_id

                                        df_train = pd.concat([df_train, SplitDataIdx[i:i + train_step]])
                                        df_val = pd.concat([df_val, SplitDataIdx[i + train_step:i + train_step + val_step]])

                                        # if len_split_data - (i + train_step + val_step) >= self.window_size:
                                        #     split_id += 1
                                        #     SplitDataIdx[i + train_step + val_step:len_split_data]['split_id'] = np.ones(len_split_data - (i + train_step + val_step)) * split_id
                                        #     df_train = pd.concat([df_train, SplitDataIdx[i + train_step + val_step:len_split_data]])
                                        # continue

                        elif 80 < len_split_data < 160:
                            train_ratio=self.train_ratio_pos[1]
                            val_ratio=1- train_ratio
                            step_size = 1
                            train_step = round((len_split_data * train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)
                            if train_step < self.window_size or val_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step}")
                            print(f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}\ttest_step: {0}\tstep_size: {step_size}")
                            for i in range(0, len_split_data, train_step + val_step):
                                if len_split_data - i >= self.window_size:
                                    if len_split_data - i >= train_step + val_step:
                                        train_pos += train_step
                                        val_pos += val_step
                                        split_id += 1
                                        SplitDataIdx[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                                        split_id += 1
                                        SplitDataIdx[i + train_step:i + train_step + val_step]['split_id'] = np.ones(
                                            val_step) * split_id
                                        df_train = pd.concat([df_train, SplitDataIdx[i:i + train_step]])
                                        df_val = pd.concat(
                                            [df_val, SplitDataIdx[i + train_step:i + train_step + val_step]])
                                        continue
                        elif 64 <= len_split_data <= 80:
                            train_ratio=self.train_ratio_pos[2]
                            val_ratio=1-train_ratio
                            step_size = 1
                            train_step = round((len_split_data * train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)
                            if train_step < self.window_size or val_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step}")
                            train_pos += train_step
                            val_pos += val_step
                            print(
                                f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}\ttest_step: {0}\tstep_size: {step_size}")
                            split_id += 1
                            SplitDataIdx[:train_step]['split_id'] = np.ones(train_step) * split_id
                            split_id += 1
                            SplitDataIdx[train_step:]['split_id'] = np.ones(len_split_data - train_step) * split_id
                            df_train = pd.concat([df_train, SplitDataIdx[:train_step]])
                            df_val = pd.concat([df_val, SplitDataIdx[train_step:]])

                        elif 32 <= len_split_data < 64:
                            train_ratio=self.train_ratio_pos[3]
                            val_ratio=1-train_ratio
                            step_size = 1
                            train_step = round((len_split_data * train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)
                            if train_step < self.window_size or val_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step}")
                            train_pos += train_step
                            val_pos += val_step
                            print(f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}\ttest_step: {0}\tstep_size: {step_size}")
                            split_id += 1
                            SplitDataIdx[:train_step]['split_id'] = np.ones(train_step) * split_id
                            split_id += 1
                            SplitDataIdx[train_step:]['split_id'] = np.ones(len_split_data - train_step) * split_id
                            df_train = pd.concat([df_train, SplitDataIdx[:train_step]])
                            df_val = pd.concat([df_val, SplitDataIdx[train_step:]])

                        elif len_split_data < 32:
                            train_step = len_split_data
                            train_pos += train_step
                            split_id += 1
                            step_size = 1
                            print(
                                f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {0}\t test_step: {0}\tstep_size: {step_size}")
                            SplitDataIdx[:]['split_id'] = np.ones(train_step) * split_id
                            df_train = pd.concat([df_train, SplitDataIdx[:]])

                elif self.window_size == 12:
                    UniqueIds = df_pos['id'].unique()
                    for index, id in enumerate(UniqueIds):
                        train_ratio = 0.7
                        val_ratio = 0.2
                        test_ratio = 0.1
                        SplitDataIdx = df_pos['id'].isin([id])
                        SplitDataIdx = df_pos.loc[SplitDataIdx, :]
                        len_split_data = len(SplitDataIdx.index)
                        if len_split_data < self.window_size:
                            continue
                        SplitDataIdx['split_id'] = np.zeros(len_split_data)
                        if len_split_data >= self.window_size * 10:  # train,val test cross val yapmak için id nin en az 80 büyüklüğünde olması gerek
                            for step in range(5, 0,-1):  # dinamik step_size, id nin uzunluğuna göre step size belirliyorz.
                                if len_split_data > self.window_size * 10 * step:
                                    step_size = step
                                    break
                            train_step = round((len_split_data * train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)
                            test_step = round((len_split_data * test_ratio) / step_size)
                            if train_step < self.window_size or val_step < self.window_size or test_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step},{test_step}")

                            print(f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}\ttest_step: {test_step}\tstep_size: {step_size}")
                            for i in range(0, len_split_data, train_step + val_step + test_step):
                                if len_split_data - i > self.window_size:
                                    if len_split_data - i >= train_step + val_step + test_step:
                                        train_pos += train_step
                                        val_pos += val_step
                                        test_pos += test_step
                                        split_id += 1
                                        SplitDataIdx[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                                        split_id += 1
                                        SplitDataIdx[i + train_step:i + train_step + val_step]['split_id'] = np.ones(
                                            val_step) * split_id
                                        split_id += 1
                                        SplitDataIdx[i + train_step + val_step:i + train_step + val_step + test_step][
                                            'split_id'] = np.ones(test_step) * split_id
                                        df_train = pd.concat([df_train, SplitDataIdx[i:i + train_step]])
                                        df_val = pd.concat(
                                            [df_val, SplitDataIdx[i + train_step:i + train_step + val_step]])
                                        df_test = pd.concat([df_test, SplitDataIdx[
                                                                      i + train_step + val_step:i + train_step + val_step + test_step]])
                                        continue
                                    elif len_split_data - i >= train_step + val_step:
                                        train_pos += train_step
                                        val_pos += val_step
                                        test_pos += len_split_data - (i + train_step + val_step)
                                        split_id += 1
                                        SplitDataIdx[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                                        split_id += 1
                                        SplitDataIdx[i + train_step:i + train_step + val_step]['split_id'] = np.ones(
                                            val_step) * split_id
                                        df_train = pd.concat([df_train, SplitDataIdx[i:i + train_step]])
                                        df_val = pd.concat(
                                            [df_val, SplitDataIdx[i + train_step:i + train_step + val_step]])
                                        if len_split_data - (i + train_step + val_step) >= self.window_size:
                                            split_id += 1
                                            SplitDataIdx[i + train_step + val_step:len_split_data][
                                                'split_id'] = np.ones(
                                                len_split_data - (i + train_step + val_step)) * split_id
                                            df_test = pd.concat(
                                                [df_test, SplitDataIdx[i + train_step + val_step:len_split_data]])
                                        continue
                        elif 40 <= len_split_data < 120:
                            val_ratio = 0.3
                            train_ratio = 1 - val_ratio
                            step_size = 1
                            train_step = round((len_split_data * train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)
                            if train_step < self.window_size or val_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step}")
                            print(
                                f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}\ttest_step: {0}\tstep_size: {step_size}")
                            for i in range(0, len_split_data, train_step + val_step):
                                if len_split_data - i >= self.window_size:
                                    if len_split_data - i >= train_step + val_step:
                                        train_pos += train_step
                                        val_pos += val_step
                                        split_id += 1
                                        SplitDataIdx[i:i + train_step]['split_id'] = np.ones(train_step) * split_id
                                        split_id += 1
                                        SplitDataIdx[i + train_step:i + train_step + val_step]['split_id'] = np.ones(
                                            val_step) * split_id
                                        df_train = pd.concat([df_train, SplitDataIdx[i:i + train_step]])
                                        df_val = pd.concat(
                                            [df_val, SplitDataIdx[i + train_step:i + train_step + val_step]])
                                        continue
                        elif 30 <= len_split_data < 60:
                            val_ratio = 0.5
                            train_ratio = 1 - val_ratio
                            step_size = 1
                            train_step = round((len_split_data * train_ratio) / step_size)
                            val_step = round((len_split_data * val_ratio) / step_size)
                            if train_step < self.window_size or val_step < self.window_size:
                                raise Exception(f"Steps should not be under window_size {train_step},{val_step}")
                            train_pos += train_step
                            val_pos += val_step
                            print(
                                f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {val_step}\ttest_step: {0}\tstep_size: {step_size}")
                            split_id += 1
                            SplitDataIdx[:train_step]['split_id'] = np.ones(train_step) * split_id
                            split_id += 1
                            SplitDataIdx[train_step:]['split_id'] = np.ones(len_split_data - train_step) * split_id
                            df_train = pd.concat([df_train, SplitDataIdx[:train_step]])
                            df_val = pd.concat([df_val, SplitDataIdx[train_step:]])

                        elif len_split_data < 30:
                            train_step = len_split_data
                            train_pos += train_step
                            split_id += 1
                            step_size = 1
                            print(
                                f"id: {id}\tlen_id: {len_split_data}\ttrain_step: {train_step}\tval_step: {0}\t test_step: {0}\tstep_size: {step_size}")
                            SplitDataIdx[:]['split_id'] = np.ones(train_step) * split_id
                            df_train = pd.concat([df_train, SplitDataIdx[:]])

            if key >= 1:
                # her bir csv dosyasından kaçı traine, val'a, teste gittiğini gösteren tablo.
                print(f"train_pos: {train_pos}\tval_pos: {val_pos}")
                pos_train = train_pos - pos_csv.loc[0:key]['train_pos'].sum()
                pos_val = val_pos - pos_csv.loc[0:key]['val_pos'].sum()

                data_pos = pd.DataFrame(data={'pos': len(df_pos.index), 'train_pos': pos_train, 'val_pos': pos_val, 'pos_train_step': train_step,'pos_val_step': val_step}, index=[key])
                pos_csv = pd.concat([pos_csv, data_pos])
            else:
                data_pos = pd.DataFrame(data={'pos': len(df_pos.index), 'train_pos': train_pos, 'val_pos': val_pos,'pos_train_step': train_step,
                                              'pos_val_step': val_step}, index=[key])
                pos_csv = pd.concat([pos_csv, data_pos])
        # df_config dosyasını oluştur.
        df_config = pd.concat([neg_csv, pos_csv], axis=1)
        df_config = df_config[['Dataset_Name', 'pos', 'train_pos', 'val_pos', 'neg', 'train_neg', 'val_neg','pos_train_step', 'pos_val_step']]
        data_path = self.path + "\\data"
        try:
            os.mkdir(data_path)
        except Exception as e:
            print(e)

        train_neg = len(df_train[df_train['label'] == 0].index)
        val_neg = len(df_val[df_val['label'] == 0].index)

        print(f"Splitten önce toplam NEGATİF sayisi: {total_neg}")
        print(
            f"Splitten sonra NEGATİF dağılımı: neg_train: {train_neg} neg_val: {val_neg} neg_total: {train_neg + val_neg}")

        total = pd.DataFrame({'Dataset_Name': 'total', 'pos': total_pos, 'train_pos': train_pos, 'val_pos': val_pos, 'neg': total_neg, 'train_neg': train_neg, 'val_neg': val_neg,
                              'pos_train_step': 0, 'pos_val_step': 0}, index=[key + 1])
        df_config = pd.concat([df_config, total])
        df_config['pos_train_ratio']=np.zeros(key+2)
        df_config.loc[:len(self.train_ratio_pos)-1,'pos_train_ratio']=self.train_ratio_pos
        df_config['neg_train_ratio']=np.zeros(key+2)
        df_config.loc[:len(self.train_val_ratio_neg)-1,'neg_train_ratio']=self.train_val_ratio_neg
        df_config.to_csv(data_path + "\\df_config.csv")
        df_train.to_csv(data_path + "\\train.csv")
        df_val.to_csv(data_path + "\\val.csv")

        # model parametrelerini datanın içinede kaydet.
        model_param = pd.DataFrame()
        model_param.loc[:, 'Feature_List'] = self.feature_columns
        model_param['Window_Size'] = self.window_size
        model_param['Stride_Size'] = self.stride
        model_param.to_csv(data_path + "\\config.csv")

        # veri dağılımı gösteren pasta grafiğini çiz.
        total = [train_pos + train_neg, val_pos + val_neg]
        neg_pos = [train_pos, train_neg, val_pos, val_neg]
        save_plots.distribution_pie(total, neg_pos, data_path)

        # TODO : 16 sampledan az değerleri pastaya ekleme
        return df_train, df_val
