import enum
from pickletools import optimize
from pydoc import classname
from sympy import sequence
import torch
import os, random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torchtext.datasets import WikiText2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Helper(nn.Module):

    def __init__(self, input_length, output_length, hidden_size, num_layers):
        super(Helper, self).__init__()
        self.embedding = nn.Embedding(input_length, input_length)
        self.rnn = nn.LSTM(input_length=input_length, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_length)
    @classmethod
    def forward(self, input, hidden):
        embedding = self.embedding(input)
        output, hidden = self.rnn(embedding, hidden)
        output = self.decoder(output)
        return output, (hidden[0].detach(), hidden[1].detach())
    

class MyModel():
    @classmethod
    def load_test_data(cls, fname):
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


    @classmethod 
    def save(self, save_path, rnn):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')
        torch.save(rnn.state_dict(), save_path)
    

    @classmethod
    def run_train(self, work_dir):
        hidden_size = 512
        sequence_len = 100
        layer_num = 3
        epochs = 75
        output_seq_len = 3
        learning_rate = 0.002
        save_path = ""
        train_iter = WikiText2(split="train")
        save_path = os.path.join(work_dir, "wikitrain.pth")
        total_data = ""
        while True:
            try:
                curr_text = next(train_iter)
                total_data += curr_text
            except:
                break
        all_chars = sorted(list(set(total_data)))
        total_data = total_data[:200]
        data_size = len(total_data)
        vocab_size = len(all_chars)
        char_to_index = { ch:i for i,ch in enumerate(all_chars) }
        index_to_char = { i:ch for i,ch in enumerate(all_chars) }

        # convert the data to index
        data_split_by_char = list(total_data)
        for i, char in enumerate(data_split_by_char):
            data_split_by_char[i] = char_to_index[char]
        
        index_to_char_data = data_split_by_char
        # set data for the CPU or GPU
        index_to_char_data = torch.tensor(index_to_char_data).to(device)
        index_to_char_data = torch.unsqueeze(index_to_char_data, dim=1)

        rnn = Helper(vocab_size, vocab_size, hidden_size, layer_num).to(device)

        # loss function optimizer 
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

        for iteration in range(1, epochs + 1): 
            info_ptr = np.random.randint(100)
            n = 0
            running_loss = 0
            hidden_state = None

            while True:
                input_sequence = index_to_char_data[info_ptr : info_ptr+sequence_len]
                target_sequence = index_to_char_data[info_ptr+1 : info_ptr+sequence_len+1]

                # pass forward
                output, hidden_state = rnn(input_sequence, hidden_state)
                
                # loss
                loss = loss_fn(torch.squeeze(output), torch.squeeze(target_sequence))
                running_loss += loss.item()

                # compute gradient 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                info_ptr += 1
                n += 1

                if info_ptr + sequence_len > 1 + data_size:
                    break
            
            info_ptr = 0
            hidden_state = None
            print("iteration: " + str(iteration))
            self.save(save_path, rnn)

            rand_index = np.random.randint(data_size-1)
            input_seq = index_to_char_data[rand_index : rand_index+1]

            print("----------------------------------------")
            while True:
                # forward pass
                output, hidden_state = rnn(input_seq, hidden_state)
                
                # construct categorical distribution and sample a character
                output = F.softmax(torch.squeeze(output), dim=0)
                dist = Categorical(output)
                index = dist.sample()
                
                # print the sampled character
                print(index_to_char[index.item()], end='')
                
                # next input is current output
                input_seq[0][0] = index.item()
                data_ptr += 1
                
                if data_ptr > output_seq_len:
                    break
                
            print("\n----------------------------------------")

    @classmethod
    def run_pred(self, work_dir):
        output_len = 3
        hidden_size = 512
        layer_num = 3

        train_path = os.path.join(work_dir, "wikitrain.pth")
        data_path = os.path.join(work_dir, "answer.txt")
        data = open(data_path, 'r').read()
        chars = sorted(list(set(data)))
        data_per_line = data.splitlines()
        
        char_to_index = { ch:i for i,ch in enumerate(chars) }
        index_to_char = { i:ch for i,ch in enumerate(chars) }

        data = list(data)
        for i, ch in enumerate(data):
            data[i] = char_to_index[ch]

        data = torch.tensor(data).to(device)
        data = torch.unsqueeze(data, dim=1)

        data_size, vocab_size = len(data), len(chars)
        rnn = Helper(vocab_size, vocab_size, hidden_size, layer_num).to(device)
        rnn.load_state_dict(torch.load(train_path, map_location=torch.device('cpu')))

        hidden_state = None
        info_ptr = 0


        for line in data_per_line:
            temp_line = line[:len(line)-3]
            input_seq = data[len(temp_line): len(temp_line) + 1]
            _, hidden_state = rnn(temp_line, hidden_state)
            input_seq = data[len(temp_line) + 1: len(temp_line) + 2]
            while True:
                
                if info_ptr  > output_len:
                    break
                output, hidden_state = rnn(line, hidden_state)

                output = F.softmax(torch.squeeze(output), dim=0)
                dist = Categorical(output)
                index = dist.sample().item()
                print(index_to_char[index], end='')
                input_seq[0][0] = index
                data_ptr += 1



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
        # train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(args.work_dir)
        print('Saving model')
        # model.save(args.work_dir) save performed in train()
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
