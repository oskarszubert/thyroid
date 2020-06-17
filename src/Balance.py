import pandas as pd
import numpy as np
from sklearn.utils import resample

def df_shape(df):
    return df.shape

def balance_dataset(df, df_y):
    df.insert(0, "value", df_y, True) 

    single_label_df_list = []
    class_labels = set(df['value']) 
    for label in class_labels:
        tmp_df = df[df['value'] == label]
        single_label_df_list.append( tmp_df )
    df = df.iloc[0:0]

    single_label_df_list.sort(key=df_shape,reverse=True)
    n_samples = single_label_df_list[0].shape[0]
    tmp_list = [ single_label_df_list[0] ]

    for tmp_df in single_label_df_list[1:]:
        tmp_list.append(resample(   tmp_df, 
                                    replace=True,            # sample with replacement
                                    n_samples=n_samples,     # to match majority class
                                    random_state=np.random)   )        # reproducible results
    single_label_df_list = tmp_list

    for tmp_df in single_label_df_list:
        df = pd.concat( [df, tmp_df] )

    X = df.drop(columns=['value'])
    Y = df['value'].values

    return X, Y
