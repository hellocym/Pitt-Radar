# copy the first half npy files inside ./radar_numpy to ./radar_numpy_train
# and the others to ./radar_numpy_test

import os
import shutil
import random

npy_path = 'radar_numpy'
train_path = 'radar_numpy_train'
val_path = 'radar_numpy_test'

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(val_path):
    os.mkdir(val_path)
    
npys = sorted(os.listdir(npy_path))
ratio = 0.5
random.shuffle(npys)

for i in range(len(npys)):
    if i < len(npys) // 2:
        os.symlink(os.path.join(npy_path, npys[i]), os.path.join(train_path, npys[i]))
    else:
        os.symlink(os.path.join(npy_path, npys[i]), os.path.join(val_path, npys[i]))