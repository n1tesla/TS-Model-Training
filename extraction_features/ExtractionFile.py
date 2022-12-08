"Import Libraries"
import pandas as pd
import numpy as np

"Define Feature Extraction Class"
class FeatureExt:

    def __init__(self, RawData):

        self.ExtractedData = pd.DataFrame()
        self.RawData=RawData

    def Main(self):

        " Get PreDefined Data "
        RawData = self.RawData
        ExtractedData = self.ExtractedData

        " Assign and Initialize Features"
        ExtractedData["id"] = RawData['ID_labeled']
        ExtractedData["azimuth"] = RawData['AZIMUTH_labeled']
        ExtractedData["elevation"] = RawData['ELEVATION_labeled']
        ExtractedData["range"] = RawData['RANGE_labeled']
        ExtractedData["heading"] = np.rad2deg(np.arctan2((RawData['VELY_labeled']), (
        RawData['VELX_labeled'])))  # atan2'ye çevrilecek. Bölheye göre değerlendirip radyan olarak veriyor.
        ExtractedData["radialVel"] = (((RawData['X_labeled'] * RawData['VELX_labeled']) + (RawData['Y_labeled'] * RawData['VELY_labeled']) +
                                       ( RawData['Z_labeled'] * RawData['VELZ_labeled'])) /
                                      (np.sqrt(RawData['X_labeled'] ** 2 + RawData['Y_labeled'] ** 2 + RawData['Z_labeled'] ** 2)))
        ExtractedData["rcs"] = RawData['RCSEst_labeled']
        ExtractedData["rcs_dsbm"] = RawData["RCSEst_DBSM_labeled"]
        ExtractedData["track_time"] = RawData['Radar_Time_labeled']
        ExtractedData["time_diff"] = RawData['TS_diff']
        ExtractedData["SystemTime"] = RawData['RadarSysTime_labeled']
        ExtractedData["pos_x"] = RawData['X_labeled']
        ExtractedData["pos_y"] = RawData['Y_labeled']
        ExtractedData["pos_z"] = RawData['Z_labeled']
        ExtractedData["vel"] = np.sqrt(
            (RawData['VELX_labeled']) ** 2 + (RawData['VELY_labeled']) ** 2 + (RawData['VELZ_labeled']) ** 2)
        ExtractedData["vel_xy"] = np.sqrt((RawData['VELX_labeled']) ** 2 + (RawData['VELY_labeled']) ** 2)
        ExtractedData["vel_x"] = RawData['VELX_labeled']
        ExtractedData["vel_y"] = RawData['VELY_labeled']
        ExtractedData["vel_z"] = RawData['VELZ_labeled']
        ExtractedData["angular_vel"]= ""
        ExtractedData["acc"] = ""
        ExtractedData["acc_xy"] = ""
        ExtractedData["acc_x"] = ""
        ExtractedData["acc_y"] = ""
        ExtractedData["acc_z"] = ""
        ExtractedData["disp"] = ""
        ExtractedData["disp_xy"] = ""
        ExtractedData["max_velocity"] = ""
        ExtractedData["probUAV"] = RawData['P_UAV_labeled']
        ExtractedData["label"] = RawData['label']

        "Initialize some features"
        ExtractedData = self.InitSomeFeatures(ExtractedData)

        " Calculate Extracted Features"
        for i in range(1,ExtractedData.shape[0]):
            ExtractedData["acc"].iloc[i] = self.getDerivative(ExtractedData, "vel", i)
            ExtractedData["acc_xy"].iloc[i] = self.getDerivative(ExtractedData, "vel_xy", i)
            ExtractedData["acc_x"].iloc[i] = self.getDerivative(ExtractedData, "vel_x", i)
            ExtractedData["acc_y"].iloc[i] = self.getDerivative(ExtractedData, "vel_y", i)
            ExtractedData["acc_z"].iloc[i] = self.getDerivative(ExtractedData, "vel_z", i)
            ExtractedData["disp_xy"].iloc[i] = self.calculateDisp(ExtractedData, i, 2)
            ExtractedData["disp"].iloc[i] = self.calculateDisp(ExtractedData, i, 3)
            ExtractedData["max_velocity"].iloc[i] = np.max(ExtractedData["vel"].iloc[:i+1].values)
            ExtractedData["angular_vel"].iloc[i] = self.calculateAngVel(ExtractedData, i)

        # Heading is regulated between -180 and +180
        ExtractedData = self.regulateHeading(ExtractedData, 'heading')

        return ExtractedData

    " Conversion Functions "

    def InitSomeFeatures(self,ExtractedData):
        ExtractedData["max_velocity"].iloc[0] = ExtractedData["vel"].iloc[0]
        ExtractedData["angular_vel"].iloc[0] = 0
        ExtractedData["acc"].iloc[0] = 0
        ExtractedData["acc_xy"].iloc[0] = 0
        ExtractedData["acc_x"].iloc[0] = 0
        ExtractedData["acc_y"].iloc[0] = 0
        ExtractedData["acc_z"].iloc[0] = 0
        ExtractedData["disp"].iloc[0] = 0
        ExtractedData["disp_xy"].iloc[0] = 0
        ExtractedData["max_velocity"].iloc[0] = ExtractedData["vel"].iloc[0]
        return ExtractedData

    def getDerivative(self, Data, columnName, k): # use the function to calculate acceleration
        dataDiff=(Data[columnName].iloc[k]-Data[columnName].iloc[k-1])
        return dataDiff/Data["time_diff"].iloc[k]

    def calculateDisp(self,Data, k, dimension):
        PosX_diff=(Data["pos_x"].iloc[k]-Data["pos_x"].iloc[k-1])
        PosY_diff=(Data["pos_y"].iloc[k]-Data["pos_y"].iloc[k-1])
        if dimension==3:
            PosZ_diff=(Data["pos_z"].iloc[k]-Data["pos_z"].iloc[k-1])
            return  np.sqrt(PosX_diff**2 + PosY_diff**2 + PosZ_diff**2)
        else:
            return  np.sqrt(PosX_diff**2 + PosY_diff**2)

    def calculateAngVel(self, Data, k):
        headingDiff = (Data["heading"].iloc[k]-Data["heading"].iloc[k-1])
        timeDiff = Data["time_diff"].iloc[k]
        if abs(headingDiff) < 250:
            return headingDiff/timeDiff
        elif ((headingDiff >= 250) and (headingDiff <= 360)):
            return (headingDiff - 360) / timeDiff
        else:
            return (headingDiff + 360) / timeDiff

    def regulateHeading(self, Data, columnName):

        for k in range(10000):
            ind=list()
            sign=list()

            # find passing of headings
            for i in range(0,Data[columnName].shape[0]-1):
                val = Data[columnName].iloc[i + 1] - Data[columnName].iloc[i]
                if abs(val)>300:
                    ind.append(i+1)
                    sign.append(np.sign(val))
                    if len(ind)==2:
                        break

            if not bool(ind):
                break

            if len(ind)==1:
                if sign[0]>0:
                    Data[columnName].iloc[ind[0]::] = Data[columnName].iloc[ind[0]::] - 360
                else:
                    Data[columnName].iloc[ind[0]::] = Data[columnName].iloc[ind[0]::] + 360
            else:
                if sign[0]>0:
                    Data[columnName].iloc[ind[0]:ind[0+1]] = Data[columnName].iloc[ind[0]:ind[0+1]] - 360
                else:
                    Data[columnName].iloc[ind[0]:ind[0+1]] = Data[columnName].iloc[ind[0]:ind[0+1]] + 360

        return Data


