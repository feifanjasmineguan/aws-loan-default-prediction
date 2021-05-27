"""
This file remove missing rows, preprocess nominal and numerical columns, 
and perform feature engineering on the origination dataset on an 
EC2 instance. 

Author: Jasmine Guan and Sheng Yang
"""

import os 
# import multiprocessing as mp
from distributed import Client
import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import OneHotEncoder

# constants
data_path = 'data/historical_data_2009Q1'
output_path = 'output/feature.parquet'

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
     

def drop_missing_origination(ddf):
    """
    remove missing entires

    :param ddf: the dask dataframe read in 
    """
    # TODO: do we need to impute any of these or other columns?
    return ddf.loc[(ddf['CREDIT_SCORE'] <= 850) &
                   (ddf['CREDIT_SCORE'] >= 301) &
                   (ddf['FIRST_TIME_HOMEBUYER_FLAG'] != '9') &
                   (ddf['NUMBER_OF_UNITS'] < 5) &
                   (ddf['OCCUPANCY_STATUS'] != '9') &
                   (ddf['DTI'] != 999) &
                   (ddf['CHANNEL'] != '9') &
                   (ddf['LOAN_PURPOSE'] != '9')
                   ]


def preprocess_origination(ddf_dropna):
    """
    preprocess: scale numeric data, convert booleans,
    and one hot encode nominal data 
    
    :param ddf: the dask dataframe read in
    """
    # preprocess numeric and boolean columns 
    ddf_num_processed = ddf_dropna.assign(
        CREDIT_SCORE=((ddf_dropna['CREDIT_SCORE'] - 301) / (850 - 301)).astype('float32'),  # don't need many spaces
        FIRST_TIME_HOMEBUYER_FLAG=ddf_dropna['FIRST_TIME_HOMEBUYER_FLAG'].apply(lambda x: x == 'Y', meta=('int')),
        HARP_INDICATOR=ddf_dropna['HARP_INDICATOR'].apply(lambda x: x == 'Y', meta=('int'))
    )

    # preprocess nominal columns 
    nom_cols = ['OCCUPANCY_STATUS', 'CHANNEL', 'LOAN_PURPOSE']
    enc = OneHotEncoder(sparse=False, dtype=int)
    nom_cols_transformed = enc.fit_transform(ddf_num_processed[nom_cols].categorize())
    ddf_proprocessed = dd.concat([ddf_num_processed, nom_cols_transformed], axis=1, ignore_unknown_divisions=True)

    # drop original nominal columns 
    return ddf_proprocessed.drop(columns=nom_cols)
    

def engineer_origination_feature(ddf_dropna):
    """
    perform feature engineering on the dropna dataframe 
    TODO: add comments on what columns to transform

    :param ddf_dropna: the dask dataframe with missing value dropped
    """
    # log product of interest rate and loan term 
    log_rate_prod_term = da.log(ddf_dropna['ORIGINAL_INTEREST_RATE'] / 100 + 1) * ddf_dropna['ORIGINAL_LOAN_TERM']
    log_rate_prod_term.name = 'LOG_RATE_PROD_TERM'  # renaming for appending 
    # log scaled UPB 
    log_UPB = da.log(ddf_dropna['ORIGINAL_UPB'])
    
    return dd.concat([log_rate_prod_term, log_UPB], axis=1, ignore_unknown_divisions=True)


def initialize_client():
    """ initialize client """
    client = Client()

# def wrap_preprocess(ddf, q):
#     q.put(preprocess_origination(ddf))

# def wrap_engineer(ddf, q):
#     q.put(engineer_origination_feature(ddf))

def main():
    """
    wrap the entire process 
    """
    ddf = read_origination(os.path.join(data_path, 'historical_data_2009Q1.txt'))
    ddf_dropna = drop_missing_origination(ddf)  # drop missing data

    # TODO: parallelize preprocess and feature engineering, maybe?
    # preprocess and feature engineering
    ddf_preprocessed = preprocess_origination(ddf_dropna).drop(columns=['ORIGINAL_UPB'])
    ddf_engineered = engineer_origination_feature(ddf_dropna)
    # combine and write to parquet
    dd.concat([ddf_preprocessed, ddf_engineered], 
              axis=1, 
              ignore_unknown_divisions=True
    ).to_parquet(output_path, overwrite=True)


# run the following code line by line in interactive python on EC2

if __name__ == '__main__':
    initialize_client()
    main()
