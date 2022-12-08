
import pandas as pd
import os,glob

dataset_path=r"C:\Users\tunahan.akyol\Desktop\lstm\workspace\matched"

df_comparision_table=pd.DataFrame(columns=['dataset_time','df_pos','df_neg','df_total','df_fus','missing_percentage'])

for dirname, filenames,_ in os.walk(dataset_path):
    for filename in filenames:
        time_path=os.path.join(dirname,filename)
        os.chdir(time_path)
        df_number_of_data=pd.DataFrame()
        for csv_files in glob.glob("*.csv"):
            if csv_files.startswith("Drone_Class"):
                df_pos=pd.read_csv(os.path.join(time_path,csv_files))
            if csv_files.startswith("Fuzyon"):
                df_fus=pd.read_csv(os.path.join(time_path,csv_files))
            if csv_files.startswith('Neg_Class'):
                df_neg=pd.read_csv(os.path.join(time_path,csv_files))
        missing_percentage=f"%{((len(df_fus.index)-(len(df_pos.index)+len(df_neg.index)))/len(df_fus.index))*100}"
        df_data=pd.DataFrame(data={"dataset_time":filename,"df_pos":len(df_pos.index),"df_neg":len(df_neg.index),
                                   "df_total":len(df_pos.index)+len(df_neg.index),"df_fus":len(df_fus.index),
                                   "missing_percentage":missing_percentage},
                             index=[0])
        df_comparision_table=pd.concat([df_comparision_table,df_data],ignore_index=True,axis=0)

df_comparision_table.to_csv(dataset_path+"\\comparision_table.csv")


