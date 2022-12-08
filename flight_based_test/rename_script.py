import os
import pandas as pd

path=r"C:\Users\tunahan.akyol\Desktop\ihtar_data_annotation_tool\Test\Data\ACAR_records\IHTAR_20220701\Labeled_Data_Out"
all_concat=pd.DataFrame()
for i in os.listdir(path):
    file=path+"\\"+i

    for csv_files in os.listdir(file):
        if csv_files.startswith('Drone'):
            pos_df=pd.read_csv(file+"\\"+csv_files)
            pos_df.to_csv(file+"\\acar_pos.csv")
        elif csv_files.startswith('Neg'):
            neg_df=pd.read_csv(file+"\\"+csv_files)
            neg_df.to_csv(file+"\\acar_neg.csv")
        elif csv_files.startswith('All'):

            all_df=pd.read_csv(file+"\\"+csv_files)
            all_concat = pd.concat([all_concat,all_df ])
            all_concat.to_csv(file+"\\acar_all.csv")



