
'''read word embedding test.'''

__author__ = 'innerpeace'

import sys
sys.path.append("..")
from Config import Config
from train import initialize_vocab
from os.path import join as pjoin
import numpy as np

data_dir = Config.DATA_DIR

def embed_test(print_out = 10):
    vocab_path = pjoin(data_dir, 'vocab.dat')
    vocab, rev_vocab = initialize_vocab(vocab_path)
    # print('first {} of vocab: {}'.format(print_out, ''.join(vocab[:print_out])))
    print('first {} of rev_vocab: {}'.format(print_out, rev_vocab[:print_out]))
    # we use glove with 100-d in default.
    embed_path = pjoin(data_dir, "glove.trimmed.100.npz")
    embedding = np.load(embed_path)
    for k, v in vocab.items()[:10]:
        print(k, v)

    print('length of embedding is {}'.format(embedding['glove'].shape))
    print('length of vocab is {}'.format(len(rev_vocab)))
    # for k, v in embedding.items()[:10]:
    #     print("------- embeddings -------")
    #     print(k, v)

if __name__ == '__main__':
    embed_test()
