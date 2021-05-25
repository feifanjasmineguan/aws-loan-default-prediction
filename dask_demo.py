# Dask Demo

from random import randint
from time import time,sleep

from dask import delayed, compute
from dask.distributed import Client