import pickle
import numpy as np
from pathlib import Path

CUR_DIR = Path(__file__).parent

f = open(CUR_DIR / "./franka_lift_cube_image_20_trajs.pkl",'rb')
data = pickle.load(f)
print(data)