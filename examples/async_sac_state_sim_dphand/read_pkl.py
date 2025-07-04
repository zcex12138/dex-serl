import pickle
import numpy as np
from pathlib import Path

CUR_DIR = Path(__file__).parent

f = open(CUR_DIR / "./dphand_20_demos_2025-06-18_17-12-34.pkl",'rb')
data = pickle.load(f)

cnt = 0
for i, transition in enumerate(data):
    if transition['dones']:
        cnt += 1
        print(f"Trajectory {cnt} is done.")
f.close()