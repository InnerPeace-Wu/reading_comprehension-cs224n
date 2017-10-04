import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import Config

cfg = Config.Config()
ROOT_DIR = cfg.ROOT_DIR
Data_dir = 'data/squad'
# suffixes = ['context', 'question', 'answer']
suffixes = ['context', 'question']
set_names = ['train', 'val']

def read_ids(set_name, suffix):
    file_name = set_name + '.ids.' + suffix
    file_path = pjoin(ROOT_DIR, Data_dir, file_name)
    file_path = os.path.abspath(file_path)

    #print(os.path.realpath(__file__))
    #print(os.path.abspath(__file__))

    assert os.path.exists(file_path), "the file path: {} seems not exist.".format(file_path)

    count_list = []
    with open(file_path, 'r') as f:
        # the lines below will read the whole file, that'a bad.
        # for line in f.readlines():
        #     count_list.append(len(line.split(' ')))
        line = f.readline()
        while line:
            count_list.append(len(line.split(' ')))
            line = f.readline()

    return count_list
    #hist_plot(count_list, set_name, suffix)
    #print(count_list)

def hist_plot(count_list, suffix, set_name='allset'):

    plt.hist(count_list, normed=True, bins=50)
    plt.ylabel('Counts')
    plt.xlabel('Length')
    plt.title('Histgrams of {}_{}'.format(set_name, suffix))
    #plt.show()

def draw_hists():
    l = len(suffixes)
    fig, ax = plt.subplots(nrows=1, ncols=l,figsize=(10,5))
    for i in xrange(l):
        count_list = []
        for sn in set_names:
            count_list += read_ids(sn, suffixes[i])
        plt.subplot(1, l, i+1)
        hist_plot(count_list, suffixes[i])
        if suffixes[i] == 'context': le = 600
        else: le = 30
        ax[i].set_xlim(0, 0.25)
        print(ax[i].get_xlim())
    fig.tight_layout()
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # plt.savefig('temp.png',dpi=100)
    plt.show()
    # to gain resolution by set dpi
    # output_path = 'histgrams-of-dataset.pdf'
    # plt.savefig(output_path, format='pdf')



if __name__ == '__main__':
    #read_ids('val', 'question')
    draw_hists()

