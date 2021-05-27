"""
On an EMR, label the monthly performance data using Spark 

Author: Jasmine Guan and Sheng Yang
"""

# testing locally ...
import os
from pyspark.sql import SparkSession

# constants 
data_path = 'data/historical_data_2009Q1'

label_cols_idx = [0, 3, 8]
label_names = ['LOAN_SEQUENCE_NUMBER', 'CURRENT_LOAN_DELINQUENCY_STATUS', 'ZERO_BALANCE_CODE']
label_types = ['str', 'str', 'str']


def read_monthly_performance(spark, file_path):
    """
    read in the monthly performance data 

    :param spark: the initialized spark
    :param file_path: the path to the txt 
    """
    return spark.read.csv(file_path, sep='|', header=None)


if __name__ == '__main__':
    # initialize spark 
    spark = SparkSession.builder.appName("Assign_Label").getOrCreate()  # give name Assign_label

    # read in data 
    df = read_monthly_performance(spark, os.path.join(data_path, 'historical_data_time_2009Q1.txt'))

    # rename columns 
    relevant_df = df.selectExpr(['_c0 as LOAN_SEQUENCE_NUMBER', 
                                 '_c3 as CURRENT_LOAN_DELINQUENCY_STATUS',
                                 '_c8 as ZERO_BALANCE_CODE'] 
                                )

    # THE FOLLOWING ARE STILLED BEING TESTED: NEED AMENDMENTS 
    relevant_df_drop_R = relevant_df.filter(relevant_df.CURRENT_LOAN_DELINQUENCY_STATUS != 'R')
    # convert to int
    relevant_df_drop_R.CURRENT_LOAN_DELINQUENCY_STATUS = relevant_df_drop_R.CURRENT_LOAN_DELINQUENCY_STATUS.cast('int')

    # groupby 
    # groupby_df = relevant_df_drop_R.groupby('LOAN_SEQUENCE_NUMBER').agg(
    #     {'CURRENT_LOAN_DELINQUENCY_STATUS': lambda x: (x > 3).any(), 
    #      'ZERO_BALANCE_CODE': lambda x: x.isin(['03', '06', '09']).any()
    #      })
    # groupby_df.show()
