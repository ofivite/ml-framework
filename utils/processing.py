import pandas as pd
import h5py

def fill_placeholders(string, placeholder_to_value):
    for placeholder, value in placeholder_to_value.items():
        string = string.replace(placeholder, str(value))
    return string

def read_hdf(file_name, key_list):
    f = h5py.File(file_name, 'r')
    f_keys = list(f.keys())
    f.close()

    df_list = []
    for k in key_list: # read specified groups into pandas
        if k in f_keys:
            df_list.append(pd.read_hdf(file_name, key=k))
        else:
            print(f'\n[WARNING] Coudln\'t find group ({k}) in file ({file_name}), will skip it\n')
            
    assert all([df_list[0].index.equals(df.index) for df in df_list[1:]]) # check that indices match between dataframes
    df = pd.concat(df_list, axis=1) # combine groups together

    return df