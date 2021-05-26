"""
feature_prep.py

The first part: preprocess and feature engineer
Author: Jasmine Guan and Sheng Yang
"""

# testing locally 
import os 
import dask.dataframe as dd
from dask_ml.preprocessing import OneHotEncoder

# constants
data_path = 'data/historical_data_2009Q1'
feature_cols_idx = [0, 2, 6, 7, 9, 10, 12, 13, 19, 20, 21, 28]
feature_name = ['CREDIT_SCORE', 
                'FIRST_TIME_HOMEBUYER_FLAG', 
                'NUMBER_OF_UNITS',
                'OCCUPANCY_STATUS',
                'DTI',
                'ORIGINAL_UPB',
                'ORIGINAL_INTEREST_RATE',
                'CHANNEL', 
                'LOAN_SEQUENCE_NUMBER',  # the unique identifier
                'LOAN_PURPOSE',
                'ORIGINAL_LOAN_TERM',
                'HARP_INDICATOR'  # is HARP or not
                ]
feature_type = ['int', 'str', 'int', 'str', 'int', 'int', 'float', 'str', 'str', 'str', 'int', 'str']


def read_origination(file_path):
    """
    read in the origination data, return the dask dataframe 

    :param file_path: the path to the origination data
    """
    df = dd.read_csv(file_path,
                     sep='|',
                     header=None,
                     usecols=feature_cols_idx,
                     dtype=dict(zip(feature_cols_idx, feature_type))
                    )
    return df.rename(columns=dict(zip(df.columns, feature_name)))  # change name
     

def preprocess_origination(df):
    """
    preprocess
    
    :param df: the dask dataframe read in
    """
    # drop missing values 
    # TODO: do we need to impute any of these or other columns?
    df_dropna = df.loc[(df['CREDIT_SCORE'] <= 850) &
                       (df['CREDIT_SCORE'] >= 301) &
                       (df['FIRST_TIME_HOMEBUYER_FLAG'] != '9') &
                       (df['NUMBER_OF_UNITS'] < 5) &
                       (df['OCCUPANCY_STATUS'] != '9') &
                       (df['DTI'] != 999) &
                       (df['CHANNEL'] != '9') & 
                       (df['LOAN_PURPOSE'] != '9')
                       ]
    # preprocess numeric and boolean columns 
    df_num_processed = df_dropna.assign(
        CREDIT_SCORE=((df['CREDIT_SCORE'] - 301) / (850 - 301)).astype('float16'),    
        FIRST_TIME_HOMEBUYER_FLAG=df['FIRST_TIME_HOMEBUYER_FLAG'].apply(lambda x: x == 'Y', meta=('int')),
        HARP_INDICATOR=df['HARP_INDICATOR'].apply(lambda x: x == 'Y', meta=('int'))
    )

    # preprocess nominal columns 
    nom_cols = ['OCCUPANCY_STATUS', 'CHANNEL', 'LOAN_PURPOSE']
    enc = OneHotEncoder(sparse=False, dtype=int)
    nom_cols_transformed = enc.fit_transform(df_num_processed[nom_cols].categorize())
    df_proprocessed = dd.concat([df_num_processed, nom_cols_transformed], axis=1, ignore_unknown_divisions=True)

    # drop original nominal columns 
    return df_proprocessed.drop(columns=nom_cols)
    


if __name__ == '__main__':
    df = read_origination(os.path.join(data_path, 'historical_data_2009Q1.txt'))
    df = preprocess_origination(df)
    print(df.compute())
