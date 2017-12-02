
'''visulize the dataset which is helpful to choose the max length of context and question.'''

import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
# from Config import Config as cfg
from config import cfg

ROOT_DIR = cfg.ROOT_DIR
Data_dir = 'data/squad'
suffixes = ['context', 'question']
set_names = ['train', 'val']


def read_ids(set_name, suffix):
    '''read indexs of the datsets'''

    file_name = set_name + '.ids.' + suffix
    file_path = pjoin(ROOT_DIR, Data_dir, file_name)
    file_path = os.path.abspath(file_path)

    assert os.path.exists(file_path), "the file path: {} seems not exist.".format(file_path)

    with open(file_path, 'r') as f:
        count_list = [len(line.split(' ')) for line in f.readlines()]

    return count_list


def hist_plot(count_list, suffix, set_name='allset'):
    '''plot histgrams of input list'''

    plt.hist(count_list, normed=True, bins=50)
    plt.ylabel('Counts')
    plt.xlabel('Length')
    plt.title('Histgrams of {}_{}'.format(set_name, suffix))
    # plt.show()


def draw_hists():
    '''
    draw histgram of context in training and validation datset,
    and the histgram of question in training and validation dataset.
    '''

    l = len(suffixes)
    fig, ax = plt.subplots(nrows=1, ncols=l, figsize=(10, 5))
    for i in xrange(l):
        count_list = []
        for sn in set_names:
            count_list += read_ids(sn, suffixes[i])
        plt.subplot(1, l, i + 1)
        hist_plot(count_list, suffixes[i])

    fig.tight_layout()
    # maxmize window
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    # to gain resolution by set dpi
    # plt.savefig('temp.png',dpi=100)
    plt.show()
    # save as pdf
    # output_path = 'histgrams-of-dataset.pdf'
    # plt.savefig(output_path, format='pdf')


if __name__ == '__main__':

    draw_hists()
    # for test
    #read_ids('val', 'question')
