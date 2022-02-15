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
from collections import Counter

def char_tokenizer(text):
  return [c for c in text]

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    trigram_freq_dist = nltk.FreqDist()
    bigram_freq_dist = nltk.FreqDist()
    all_chars = set({})

    @classmethod
    def load_training_data(cls):
        return

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
        train_iter = WikiText2(split='train')
        counter = 0
        while True:
            try:
                curr_text = next(train_iter)
                self.all_chars.update(set(char_tokenizer(curr_text)))
                if counter % 1000 == 0:
                    print(counter)
                counter += 1
                if len(curr_text) >= 1:
                    self.trigram_freq_dist[("<START>", "<START>", curr_text[0])] += 1 
                    self.bigram_freq_dist[("<START>", curr_text[0])] += 1 
                if len(curr_text) >= 2:
                    self.trigram_freq_dist[("<START>", curr_text[0], curr_text[1])] += 1 
                self.trigram_freq_dist = self.trigram_freq_dist + (nltk.FreqDist(ngrams(char_tokenizer(curr_text),3)))
                self.bigram_freq_dist = self.bigram_freq_dist + (nltk.FreqDist(ngrams(char_tokenizer(curr_text),2)))
                # print(trigram_freq_dist.items())
                self.bigram_freq_dist.get(curr_text[0])
            except StopIteration:
                break
            # except Exception as e:
            #     print(e) # or whatever kind of logging you want
        

    def run_pred(self, data):
        # your code here
        preds = []
        all_char_arr = []
        for i in range(0, 144697): # all unicode chars
            all_char_arr.append(chr(i))
        all_char_arr = np.array(all_char_arr)
        for inp in data:
            char_prob = np.zeros(np.shape(all_char_arr))
            if len(inp) <= 1:
                last_letter = "<START>"
            else:
                last_letter =  inp[len(inp) - 1]
            if len(inp) <= 0:
                second_last_letter = "<START>"
            else:
                second_last_letter = inp[len(inp) - 2]
            for i, letter in enumerate(all_char_arr):
                char_prob[i] = (self.trigram_freq_dist[(letter, second_last_letter, last_letter )] + 1) / (self.bigram_freq_dist[(second_last_letter, last_letter)] + 1)
            # if temp > max_prob and letter is not "<START>":
            # indices = np.argpartition(char_prob, -3)[-3:]
            indices = char_prob.argsort()[-3:][::-1]
            # this model just predicts a random character each time
            top_guesses = all_char_arr[indices]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')
        with open(os.path.join(work_dir,"training_tri.csv"), "w") as f:
            writer = csv.writer(f)
            for key, val in self.trigram_freq_dist.items():
                writer.writerow([key,val])
        
        with open(os.path.join(work_dir,"training_bi.csv"), "w") as f:
            writer = csv.writer(f)
            for key, val in self.bigram_freq_dist.items():
                writer.writerow([key,val])  

    @classmethod
    def load(self, cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        tri_csv_file = open(os.path.join(work_dir, 'training_tri.csv'), "r")
        tri_dict_reader = csv.DictReader(tri_csv_file)
        with open(os.path.join(work_dir, 'training_bi.csv'), "r") as f:
            read = csv.reader(f)
            for row in read:
                self.bigram_freq_dist[eval(row[0])] = int(row[1])
        with open(os.path.join(work_dir, 'training_tri.csv'), "r") as f:
            read = csv.reader(f)
            for row in read:
                self.trigram_freq_dist[eval(row[0])] = int(row[1])

        return MyModel()


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
