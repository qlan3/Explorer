import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from utils.sweeper import Sweeper

exp = 'minatar_me_vae'
l = [1124,996,1216,1360,928,732,720,608,1018,562,1178,846,1014,842,1290,970,511,963,507,523,1335,971,979,1339,1323,1019,489,509,989,1289,1145,1309,1153,993,1149,841]
ll = []
for r in range(10):
  for x in l:
    ll.append(x+1440*r)
ll.sort()

file_name='log.txt'
max_line_length=10000

config_file = f'./configs/{exp}.json'
sweeper = Sweeper(config_file)
# Read a list of logs
print(f'[{exp}]: ', end=' ')
for i in ll:
  log_file = f'./logs/{exp}/{i}/{file_name}'
  try:
    with open(log_file, 'r') as f:
      # Get last line
      try:
        f.seek(-max_line_length, os.SEEK_END)
      except IOError:
        # either file is too small, or too many lines requested
        f.seek(0)
      last_line = f.readlines()[-1]
      # Get time info in last line
      try:
        t = float(last_line.split(' ')[-2])
      except:
        print(i, end=', ')
        continue
  except:
    print(i, end=', ')
    continue
print()