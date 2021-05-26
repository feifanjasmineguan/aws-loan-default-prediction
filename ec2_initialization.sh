# update system
sudo apt-get update
# install aws 
sudo apt-get install awscli
sudo apt-get install tmux
sudo apt-get install python3-venv
python3 -m venv env
source env/bin/activate
pip install "dask[complete]" --no-cache-dir
 