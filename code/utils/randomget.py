import glob
import random
import numpy as np

train = 210
test = 900

demo = glob.glob('./BraTS*')
print(demo)

np.random.shuffle(demo)

for i in range(train):
    print(demo[i][2:])

print('--------------------------------')

for i in range(train, train + test):
    print(demo[i][2:])
