import sys
sys.path.append("..")
from evaluate import f1_score, exact_match_score
from utils.Config import Config as cfg
from os.path import  join as pjoin
import numpy as np

data_dir = cfg.DATA_DIR
root = cfg.ROOT_DIR
an_path = pjoin(data_dir, 'val.answer')

def eval_text():
    with open(an_path) as f:
        raw_data = [line.strip() for line in f.readlines()]
    print(raw_data[:10])

    l = ['Corpus','Juris', 'canonici']
    s = ' '.join(l)
    print(s)
    print(f1_score(s, raw_data[0]))
    print(exact_match_score(s, raw_data[0]) / 1.0)

def zip_test():
    a = [x for x in xrange(10)]
    b = [1, 2, 3]
    c = [2, 4, 6]
    print(a)
    for i, bc in enumerate(zip(b, c)):
        print(i)
        b1, c1 = bc
        print(a[b1:c1+1])

def npsave():
    file_path = pjoin(root, 'cache', 'text.npy')
    dict = {}
    dict['a'] = [x for x in xrange(10)]
    np.save(file_path, dict)

    b = np.load(file_path)
    print(b)


def ensamble():
    # file_path = pjoin(root, 'code/train/cache.npy')
    file_path = '/home/joe/git/reading_comprehension/code/train/cache.npy'
    data = np.load(file_path)
    print(data.shape)
    s = data[0]
    e = data[1]
    print(s.shape)
    a = 39
    ss = np.zeros((4000,), dtype=np.int32)
    ee = np.zeros((4000,), dtype=np.int32)

    for i in xrange(4000):
        ss[i] = navie(data[0][i])
        ee[i] = navie(data[1][i])
        # if i < 10:
        #     print(d[i])
        #     print(ss[i])

    # print(s[100:150])
    # print(ss[100:150])
    return ss, ee


def navie(a):
    for i in xrange(3):
        if a[i] in a[i+1:]:
            return a[i]
    return np.mean(a)

if __name__ == '__main__':
    # eval_text()
    # zip_test()
    # npsave()
    ensamble()