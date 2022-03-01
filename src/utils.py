import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def one_hot_encode(arr, n_labels):
    # Initialize array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Make ones array 
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    #reshape to original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
def get_batches(arr, n_seqs, n_steps):

    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size

    # make full batch
    arr = arr[:n_batches * batch_size]
    # Reshape n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n + n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


class CharRNN(nn.Module):

    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, len(self.chars))

        # initialize the weights
        self.init_weights()

    def forward(self, x, hc):

        x, (h, c) = self.lstm(x, hc)

        x = self.dropout(x)

        # Stack up LSTM outputs using view
        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)

        x = self.fc(x)

        # return x and the hidden state (h, c)
        return x, (h, c)

    def predict(self, char, h=None, cuda=False, top_k=None):
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        inputs = torch.from_numpy(x)
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        if cuda:
            p = p.cpu()

        if top_k is None:
            char = np.max(p.numpy().squeeze())
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.squeeze()
            # p = p.numpy().squeeze()
            # char = np.random.choice(top_ch, p=p/p.sum())
            char = torch.sort(top_ch)[0]

        return self.int2char[char], h

    def init_weights(self):
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())


def train(net, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, cuda=False, print_every=10):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(n_seqs)
        for x, y in get_batches(data, n_seqs, n_steps):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net.forward(inputs, h)
            loss = criterion(output, targets.view(n_seqs * n_steps))

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()

            if counter % print_every == 0:

                # Get validation loss
                val_h = net.init_hidden(n_seqs)
                val_losses = []
                for x, y in get_batches(val_data, n_seqs, n_steps):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(n_seqs * n_steps))

                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


def sample(net, size, prime='The', top_k=None, cuda=False):
    if cuda:
        net.cuda()
    else:
        net.cpu()

    net.eval()

    def predict(char, h=None, cuda=False, top_k=None):

        if h is None:
            h = net.init_hidden(1)

        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = net.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        p = p.squeeze()
        # print(p)
        result = None
        if top_k is None:
            char_idx = torch.argmax(p).item()
            result = net.int2char[char_idx]
        else:
            _, indices = torch.topk(p, k=top_k, largest=True, sorted=True)
            result = ""
            for idx in indices:
                result += (net.int2char[idx.item()])

        return result, h

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in range(len(prime) - 1):
        # print(prime[ch])
        char, h = predict(prime[ch], h, cuda=cuda, top_k=None)
        # print(char)
    result = None
    for ii in range(size):
        char, h = predict(chars[-1], h, cuda=cuda, top_k=top_k)
        result = char

    return result

