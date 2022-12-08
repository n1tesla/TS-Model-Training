import os
from configparser import ConfigParser
from sklearn import preprocessing

def make_dir(path):
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)


def config_creator(features,scaler,window_size,stride):
    config=ConfigParser()

    # for key,value in enumerate(features):
    # config['features']=features

    config.add_section("scaler_mean")
    config.add_section("scaler_std")
    for i in range(len(features)):

        config.set("scaler_mean",features[i],str(scaler.mean_[i]))
        config.set("scaler_std",features[i],str(scaler.scale_[i]))

    config['model_param'] = {'window_size': window_size, 'stride': stride}
    config["features"]={'features':features}
    return config


