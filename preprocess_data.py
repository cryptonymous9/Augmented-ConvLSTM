import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')



DIR = config.get('Paths', 'dir')
DIR_elevation = config.get('Paths', 'elevation')
DIR_humidity = config.get('Paths', 'humidity')
DIR_pressure = config.get('Paths', 'pressure')
DIR_vwind = config.get('Paths', 'vwind')
DIR_uwind = config.get('Paths', 'uwind')
DIR_omega = config.get('Paths', 'omega')
DIR_gcm = config.get('Paths', 'gcm_prec')
DIR_observed = config.get('Paths', 'observed_prec')

DIR_monsoon_gcm = config.get('Paths', 'processed_monsoon_gcm')
DIR_monsoon_observed = config.get('Paths', 'processed_monsoon_obs')
DIR_non_monsoon_gcm = config.get('Paths', 'processed_non_monsoon_gcm')
DIR_non_monsoon_observed = config.get('Paths', 'processed_non_monsoon_obs')


gcm_start_index = config.get('DataOptions', 'gcm_start_index')
observed_start_index = config.get('DataOptions', 'observed_start_index')
pressure_end_index = config.get('DataOptions', 'pressure_end_index')
rhum_end_index =  config.get('DataOptions', 'rhum_end_index')
omega_end_index = config.get('DataOptions', 'omega_end_index')
uwind_end_index = config.get('DataOptions', 'uwind_end_index')
vwind_end_index = config.get('DataOptions', 'vwind_end_index')
projection_dimensions = config.get('DataOptions', 'projection_dimensions')
channels = config.get('DataOptions', 'channels')

def convert_nc_to_numpy():
    return 0

def load_from_numpy():
    X = np.load(DIR_gcm + 'xdata.npy' )
    y = np.load(DIR_observed + 'ydata.npy' )
    omega = np.load(DIR_omega + 'omega.npy')
    pressure = np.load(DIR_pressure + 'pressure.npy')
    rhum = np.load(DIR_humidity + 'rhum.npy')
    uwnd = np.load(DIR_uwind + 'uwnd.npy')
    vwnd = np.load(DIR_vwind + 'vwnd.npy')
    elev = np.load(DIR_elevation + 'elevation.npy')
    return X, y, omega, pressure, rhum, uwnd, vwnd, elev


def adjust_data(X, y, omega, pressure, rhum, uwnd, vwnd):
    '''
    GCM and Observed prjections data:
        1920-2005 -> 1948-2005 (10220:)
    
    Auxilliary Climatic Variables: 
        1948-2018 -> 1948-2005 (: 21170)
    '''
    X = X[gcm_start_index:]
    y = y[observed_start_index:]
    omega = omega[:omega_end_index]
    pressure = pressure[:pressure_end_index]
    rhum = rhum[:rhum_end_index]
    uwnd = uwnd[:uwind_end_index]
    vwnd = vwnd[:vwind_end_index]
    return X, y, omega, pressure, rhum, uwnd, vwnd

    
def combine_data(X, y, omega, pressure, rhum, uwnd, vwnd):
    X_final = np.zeros((channels, np.max(X.shape), projection_dimensions[0], projection_dimensions[1]))
    X_final[0,] = X
    X_final[1,] = elev
    X_final[2,] = rhum
    X_final[3,] = pressure
    X_final[4,] = omega
    X_final[5,] = uwnd
    X_final[6,] = vwnd
    return X_final


def split_for_monsoon(X, X_full_ch_last):
    '''
    Non-Monsoon months -> Jan-April, Nov, Dec
    Monsoon months -> May-Oct

    '''
    X_low, y_low   = [], [] 
    X_high, y_high = [], []

    for i in range(X.shape[0]):
        day = i%365
        if day>120 and day<304:
            X_high.append(X_full_ch_last[i])
            y_high.append(y[i])
        else:
            X_low.append(X_full_ch_last[i])
            y_low.append(y[i])

    X_low, y_low = np.asarray(X_low), np.asarray(y_low)
    X_high, y_high = np.asarray(X_high), np.asarray(y_high)

    X_low, X_high = X_low.transpose(3,0,1,2), X_high.transpose(3,0,1,2)

    print("Shape Low Precipitation X: ", X_low.shape, " Y: ", y_low.shape)
    print("Shape High Precipitation X: ", X_high.shape, " Y: ", y_high.shape)
    print('------------')
    return X_low, y_low, X_high, y_high


def split_save_to_numpy(X_low, y_low, X_high, y_high):
    np.save( DIR_monsoon_gcm + 'X_low.npy',X_low)
    np.save(DIR_monsoon_observed + 'Y_low.npy',y_low)
    print("Low Precipitation Data Saved!")

    np.save( DIR_non_monsoon_gcm + 'X_high.npy',X_high)
    np.save(DIR_non_monsoon_observed + 'Y_high.npy',y_high)
    print("High Precipitation Data Saved!")
    return 0


if __name__ == "__main__":
    convert_nc_to_numpy()
    X_r, y_r, omega_r, pressure_r, rhum_r, uwnd_r, vwnd_r, elev = load_from_numpy()
    
    print("Shape of raw GCM X data: ",X_r.shape) 
    print("Shape of raw GCM y data: ",y_r.shape)
    print("Shape of each raw Auxilliary data: ",omega_r.shape)
    print('-------------')
    X, y, omega, pressure, rhum, uwnd, vwnd = adjust_data(X_r, y_r, omega_r, pressure_r, rhum_r, uwnd_r, vwnd_r)

    print("\nUpdated Shape of GCM X: ",X.shape) 
    print("Updated Shape of GCM y: ",y.shape)
    print("Updated Shape of Aux o: ",omega.shape)
    
    assert X.shape == y.shape == omega.shape == ( np.max(X.shape), projection_dimensions[0], projection_dimensions[1])

    X_full_year = combine_data(X, y, omega, pressure, rhum, uwnd, vwnd)
    X_full_ch_last = X_full_year.transpose(1,2,3,0)
    
    X_low, y_low, X_high, y_high = split_for_monsoon(X, X_full_ch_last)
    split_save_to_numpy(X_low, y_low, X_high, y_high)
    
