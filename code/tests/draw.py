import sys
sys.path.append('..')
from utils.Config import Config as cfg
from os.path import join as pjoin
import numpy as np
import json
import matplotlib.pyplot as plt

root = cfg.ROOT_DIR

def draw_test():
    data_path = pjoin(root, 'cache/data.json')
    with open(data_path) as f:
        data = json.load(f)
    a = [d[2] for d in data]
    plt.plot(filter(a))
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    print(a)

def filter(a, b=0.8):
    a0 = a[0]
    for i in xrange(1, len(a)):
        a[i] = b * a[i -1] + (1-b)*a[i]
    return a


if __name__ == '__main__':
    draw_test()
