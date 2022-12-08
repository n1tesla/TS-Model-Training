#%%
"Import Libraries"
import pandas as pd
import os
from ExtractionFile import FeatureExt
import warnings
warnings.filterwarnings('ignore')
import tkinter.filedialog


"Init Config Parameters"
DataName = "filtered_data.csv" # enter the standard file name
# Select the main data folder path; Ex: fusion_records_20
root = tkinter.Tk()
ExtractedFilePaths = tkinter.filedialog.askdirectory(initialdir=os.getcwd(), title="SELECT THE FOLDER PATH THAT INCLUDES")
root.destroy()

for root, dirs, files in os.walk(ExtractedFilePaths, topdown = True):
   for name in dirs:
      print(os.path.join(root, name))
      filepath = os.path.join(root, name)
      if os.path.exists(os.path.join(filepath, DataName)): # Run if desired data exists
         RawData = pd.read_csv(os.path.join(filepath, DataName)) # Main Data
         ExtractedData = pd.DataFrame() # Data with extracted features
         uniqueIDs = pd.unique(RawData["ID_labeled"])
         for k in range(len(uniqueIDs)):
            idData = RawData[RawData["ID_labeled"] == uniqueIDs[k]]
            FS=FeatureExt(idData)
            ExtractedData=pd.concat([FS.Main(), ExtractedData])
         ExtractedData.to_csv(os.path.join(filepath,"extractedData_All_Train.csv"))


