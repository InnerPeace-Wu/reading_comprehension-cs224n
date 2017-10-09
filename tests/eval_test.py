
'''several tests for evaluation.'''

__author__ = 'innerpeace'

import sys
sys.path.append("..")
from evaluate import f1_score, exact_match_score
from Config import Config as cfg
from os.path import  join as pjoin
from collections import Counter
import numpy as np

data_dir = cfg.DATA_DIR
root = cfg.ROOT_DIR
an_path = pjoin(data_dir, 'val.answer')

def eval_text():
    '''figure out how evaluation works.'''

    with open(an_path) as f:
        raw_data = [line.strip() for line in f.readlines()]
    print(raw_data[:10])

    l = ['Corpus','Juris', 'canonici']
    s = ' '.join(l)
    print(s)
    print(f1_score(s, raw_data[0]))
    print(exact_match_score(s, raw_data[0]) / 1.0)

def zip_test():
    '''test zip and enumearte.'''

    a = [x for x in xrange(10)]
    b = [1, 2, 3]
    c = [2, 4, 6]
    print(a)
    for i, bc in enumerate(zip(b, c)):
        print(i)
        b1, c1 = bc
        print(a[b1:c1+1])

def npsave():
    '''test use of numpy array'''

    file_path = pjoin(root, 'cache', 'text.npy')
    dict = {}
    dict['a'] = [x for x in xrange(10)]
    np.save(file_path, dict)

    b = np.load(file_path)
    print(b)


def ensemble():
    '''navie way of doing 4 models ensemle'''

    file_path = pjoin(root, 'train/cache.npy')
    data = np.load(file_path)
    print(data.shape)
    s = data[0]
    e = data[1]
    print(s.shape)
    ss = np.zeros((4000,), dtype=np.int32)
    ee = np.zeros((4000,), dtype=np.int32)

    for i in xrange(4000):
        # ss[i] = navie(data[0][i])
        ss[i] = c_counter(data[0][i])
        ee[i] = c_counter(data[1][i])
        # if i < 10:
        #     print(d[i])
        #     print(ss[i])

    # print(s[100:150])
    # print(ss[100:150])
    return ss, ee


def navie(a):
    '''save the the item showed twice or more, otherwise
    just return the mean of the list'''

    for i in xrange(3):
        if a[i] in a[i+1:]:
            return a[i]
    return np.mean(a)

def c_counter(a=[], debug=False):
    '''use collections.Counter do models ensemble.'''

    # assert isinstance(a, list or np.ndarray), 'wrong type of input'
    if debug:
        i = raw_input('input a list: ')
        a = map(int,i.strip().split(' '))
    c = Counter(a)
    r = sorted(c.items(), key=lambda x: x[1], reverse=True)
    # print(Counter(a))
    # print(r)
    return r[0][0]

if __name__ == '__main__':
    # eval_text()
    # zip_test()
    # npsave()
    ensemble()
    # c_counter(debug=True)