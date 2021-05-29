# this only records instructions for setting up an EC2 once it has been established 
# update all relevant apps 
sudo apt-get update
# install aws 
sudo apt-get install awscli
sudo apt-get install python3-venv
# install python virtual enviroment 
python3 -m venv env
source env/bin/activate
pip install pyarrow  # for recording parquet
pip install "dask[complete]" --no-cache-dir
pip install "dask-ml[xgboost]" --no-cache-dir   # also install xgboost and dask-xgboost
pip install "dask-ml[complete]" --no-cache-dir  # install all optional dependencies
