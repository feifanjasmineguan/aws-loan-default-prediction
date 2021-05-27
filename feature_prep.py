"""
feature_prep.py

The first part: preprocess and feature engineer
Author: Jasmine Guan and Sheng Yang
"""

# all things test on EC2! Good to go! (THIS LINE TO BE REMOVED)
import os 
import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import OneHotEncoder

# constants
data_path = 'data/historical_data_2009Q1'

# features to use 
# TODO: do we need to add more?
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
    ddf = dd.read_csv(file_path,
                     sep='|',
                     header=None,
                     usecols=feature_cols_idx,
                     dtype=dict(zip(feature_cols_idx, feature_type))
                    )
    return ddf.rename(columns=dict(zip(ddf.columns, feature_name)))  # change name
     

def preprocess_origination(ddf):
    """
    preprocess: remove missing entries, scale numeric data, convert booleans,
    and one hot encode nominal data 
    
    :param ddf: the dask dataframe read in
    """
    # drop missing values 
    # TODO: do we need to impute any of these or other columns?
    ddf_dropna = ddf.loc[(ddf['CREDIT_SCORE'] <= 850) &
                       (ddf['CREDIT_SCORE'] >= 301) &
                       (ddf['FIRST_TIME_HOMEBUYER_FLAG'] != '9') &
                       (ddf['NUMBER_OF_UNITS'] < 5) &
                       (ddf['OCCUPANCY_STATUS'] != '9') &
                       (ddf['DTI'] != 999) &
                       (ddf['CHANNEL'] != '9') & 
                       (ddf['LOAN_PURPOSE'] != '9')
                       ]
    # preprocess numeric and boolean columns 
    ddf_num_processed = ddf_dropna.assign(
        CREDIT_SCORE=((ddf['CREDIT_SCORE'] - 301) / (850 - 301)).astype('float32'),  # don't need many spaces
        FIRST_TIME_HOMEBUYER_FLAG=ddf['FIRST_TIME_HOMEBUYER_FLAG'].apply(lambda x: x == 'Y', meta=('int')),
        HARP_INDICATOR=ddf['HARP_INDICATOR'].apply(lambda x: x == 'Y', meta=('int'))
    )

    # preprocess nominal columns 
    nom_cols = ['OCCUPANCY_STATUS', 'CHANNEL', 'LOAN_PURPOSE']
    enc = OneHotEncoder(sparse=False, dtype=int)
    nom_cols_transformed = enc.fit_transform(ddf_num_processed[nom_cols].categorize())
    ddf_proprocessed = dd.concat([ddf_num_processed, nom_cols_transformed], axis=1, ignore_unknown_divisions=True)

    # drop original nominal columns 
    return ddf_proprocessed.drop(columns=nom_cols)
    

def engineer_origination_feature(ddf):
    """
    perform feature engineering on the preprocessed dataframe 
    TODO: add comments on what columns to transform

    :param ddf: the preprocessed dask dataframe 
    """
    # log product of interest rate and loan term 
    log_rate_prod_term = da.log(ddf['ORIGINAL_INTEREST_RATE'] / 100 + 1) * ddf['ORIGINAL_LOAN_TERM']

    # log scaled UPB 
    log_UPB = da.log(ddf['ORIGINAL_UPB'])
    
    return ddf.assign(LOG_RATE_PROD_TERM=log_rate_prod_term.astype('float32'), 
                      LOG_UPB=log_UPB.astype('float32')
                    )

# run the following code line by line in interactive python on EC2

if __name__ == '__main__':
    ddf = read_origination(os.path.join(data_path, 'historical_data_2009Q1.txt'))
    ddf_preprocessed = preprocess_origination(ddf)   # preprocess 
    ddf_engineered = engineer_origination_feature(ddf_preprocessed)  # feature engineering 
    ddf_engineered.to_parquet('output/')
