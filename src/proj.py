#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from numpy import array
from torchtext.datasets import WikiText2
import nltk
from nltk import ngrams
import numpy as np
import csv
from utils import *

def char_tokenizer(text):
    return [c for c in text]


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self):
        self.net = None
    @classmethod
    def load_training_data(cls):
        text = ""
        train_iter = WikiText2(split="train")
        while True:
            try:
                curr_text = next(train_iter)
                text += curr_text
            except:
                break
        chars = tuple(set(text))
        int2char = dict(enumerate(chars))
        char2int = {ch: ii for ii, ch in int2char.items()}
        encoded = np.array([char2int[ch] for ch in text])
        return chars,int2char,char2int,encoded

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        chars, int2char, char2int, encoded = data
        net = CharRNN(chars, n_hidden=512, n_layers=2)
        n_seqs, n_steps = 128, 100
        train(net, encoded, epochs=10, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=True, print_every=10)
        self.net = net

    def run_pred(self, data):
        # your code here
        preds = []
        for inp in data:
            top_guesses = sample(self.net, 1, prime=inp, top_k=3, cuda=True)
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        model_name = 'rnn_1_epoch.net'

        checkpoint = {'n_hidden': self.net.n_hidden,
                      'n_layers': self.net.n_layers,
                      'state_dict': self.net.state_dict(),
                      'tokens': self.net.chars}

        with open(model_name, 'wb') as f:
            torch.save(checkpoint, f)

    @classmethod
    def load(self, cls, work_dir):
        with open('rnn_1_epoch.net', 'rb') as f:
            checkpoint = torch.load(f)

        loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        loaded.load_state_dict(checkpoint['state_dict'])
        self.net= loaded


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(work_dir=args.work_dir, cls="")
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
