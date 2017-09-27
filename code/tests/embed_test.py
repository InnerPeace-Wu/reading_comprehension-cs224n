import sys
sys.path.append("..")
from utils.Config import Config
from train import initialize_vocab
from os.path import join as pjoin
import numpy as np
import tensorflow as tf

data_dir = Config.DATA_DIR

# def initialize_vocab(vocab_path):
#     if tf.gfile.Exists(vocab_path):
#         rev_vocab = []
#         with tf.gfile.GFile(vocab_path, mode="rb") as f:
#             rev_vocab.extend(f.readlines())
#         rev_vocab = [line.strip('\n') for line in rev_vocab]
#         # (word, index)
#         vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
#         return vocab, rev_vocab
#     else:
#         raise ValueError("Vocabulary file %s not found.", vocab_path)

def embed_test(print_out = 10):
    vocab_path = pjoin(data_dir, 'vocab.dat')
    vocab, rev_vocab = initialize_vocab(vocab_path)
    # print('first {} of vocab: {}'.format(print_out, ''.join(vocab[:print_out])))
    print('first {} of rev_vocab: {}'.format(print_out, rev_vocab[:print_out]))

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
