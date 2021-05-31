"""
On an EMR, label the monthly performance data using Spark 

Author: Jasmine Guan and Sheng Yang
"""

# EMR tested fine! 
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, udf
from pyspark.sql.types import BooleanType

# IO paths
s3_bucket = 's3://ds102-bubbletea-scratch'
data_path = os.path.join(s3_bucket, 'historical_data_2009Q1')
output_path = os.path.join(s3_bucket, 'preprocess/label.parquet')


def read_monthly_performance(spark, file_path):
    """
    read in the monthly performance data 

    :param spark: the initialized spark
    :param file_path: the path to the txt 
    """
    return spark.read.csv(file_path, sep='|', header=None)


# need typing to tell spark we are returning bool; this also supports | (logical or) later
@udf(returnType=BooleanType())
def check_delinquency_status(lst):
    """
    check if the CURRENT_LOAN_DELINQUeNCY_STATUS contains 3 or more (90 days or more). 
    'R' is not counted as delinquent, per Piazza 

    :param lst: the result from group by using collect_set 
    :return True if considered delinquent, False otherwise  
    """
    for e in lst:
        try: 
            status = int(e)
        except:
            status = 0
        if status >= 3:  # once detect a delinquent, return 
            return True
    return False


@udf(returnType=BooleanType()) 
def check_zero_balance_code(lst):
    """
    check if the zero balance code indicates a delinquent. 

    :param lst: the result from group by using collect_set 
    :return True if delinquent, False otherwise
    """
    if '03' in lst or '06' in lst or '09' in lst:
        return True
    return False


def assign_label(df):
    """
    give labels to the read in dataframe 

    :param df: the df read in 
    :return a dataframe with only the unique identifier and the label (1 or 0)
    """
    # rename columns
    relevant_df = df.selectExpr(['_c0 as LOAN_SEQUENCE_NUMBER',
                                '_c3 as CURRENT_LOAN_DELINQUENCY_STATUS',
                                 '_c8 as ZERO_BALANCE_CODE']
                                )

    # check individual column: collect set and then aggregate
    is_delinquent_raw = relevant_df.groupby('LOAN_SEQUENCE_NUMBER').agg(
        collect_set('CURRENT_LOAN_DELINQUENCY_STATUS').alias(
            'delinq_status_set'),
        collect_set('ZERO_BALANCE_CODE').alias('balance_code_set')
    ).withColumn(
        'delinq_status_set', check_delinquency_status('delinq_status_set')
    ).withColumn(
        'balance_code_set', check_zero_balance_code('balance_code_set')
    )

    # combine the two boolean columns to an integer column
    is_delinquent = is_delinquent_raw.withColumn(
        'label', (is_delinquent_raw.delinq_status_set |
                  is_delinquent_raw.balance_code_set).cast('integer')  # cast to integer for storage
    ).drop(
        'delinq_status_set', 'balance_code_set'  # drop columns to save space 
    )
    return is_delinquent


def main():
    """
    wrap the entire process into one function 
    """
    # initialize spark
    spark = SparkSession.builder.appName(
        "Assign_Label"
    ).getOrCreate()  # give name Assign_label

    # read in data
    df = read_monthly_performance(
        spark, os.path.join(
        data_path, 'historical_data_time_2009Q1.txt'
        )
    )

    # assign label 
    is_delinquent = assign_label(df)

    # write to parquet (overwrite)
    is_delinquent.write.format('parquet').mode('overwrite').save(output_path)

    spark.stop()


if __name__ == '__main__':
    main()
