# copy the first half npy files inside ./radar_numpy to ./radar_numpy_train
# and the others to ./radar_numpy_test

import os
import shutil

npy_path = 'radar_numpy'
train_path = 'radar_numpy_train'
val_path = 'radar_numpy_test'

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(val_path):
    os.mkdir(val_path)
    
npys = sorted(os.listdir(npy_path))

for i in range(len(npys)):
    if i < len(npys) // 2:
        shutil.copy(os.path.join(npy_path, npys[i]), os.path.join(train_path, npys[i]))
    else:
        shutil.copy(os.path.join(npy_path, npys[i]), os.path.join(val_path, npys[i]))