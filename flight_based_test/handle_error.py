import pandas as pd


path=r"C:\Users\tunahan.akyol\Desktop\lstm\data\test_fusion_dataset\06_30_13_20_zikzak"
feature_columns = ["elevation", "vel", "acc", "range","rcs_dsbm"]
df = pd.read_csv(r"C:\Users\tunahan.akyol\Desktop\lstm\data\test_fusion_dataset\06_30_13_20_zikzak\acar.csv")

try:
    df=pd.read_csv(r"C:\Users\tunahan.akyol\Desktop\lstm\data\test_fusion_dataset\06_30_13_20_zikzak\acar.csv")
except FileNotFoundError:

    df=pd.DataFrame(columns=feature_columns)
    print(df)
