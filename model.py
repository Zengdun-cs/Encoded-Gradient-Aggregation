import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import math
from torch.nn import init

# residual AE
# 128 -> 32 -> 128


class BasicBlock(nn.Module):
    def __init__(self, in_dim, expansion=1):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, in_dim * expansion),
                                   nn.ReLU(),
                                   nn.Linear(in_dim * expansion, in_dim),
                                   nn.ReLU()
                                   )

    def forward(self, x):
        y = self.layer(x)
        return x + y

class SymmetryEncoder(nn.Module):
    def __init__(self, num_block=3, expansion=4, in_dim=128, out_dim=64):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_layer = nn.Sequential(nn.Linear(self.in_dim, 512), nn.ReLU(),
                                      nn.Linear(512, 512))
        self.res_block = nn.Sequential()
        for i in range(num_block):
            self.res_block.add_module("residual" + str(1),
                                      BasicBlock(512, expansion=expansion))

        self.out_layer = nn.Sequential(nn.Linear(512, self.out_dim))

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block(x)
        x = self.out_layer(x)
        return x


class SymmetryDecoder(nn.Module):
    def __init__(self, num_block=3, expansion=4, in_dim=64, out_dim=128):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_layer = nn.Sequential(nn.Linear(self.in_dim, 512))

        self.res_block = nn.Sequential()
        for _ in range(num_block):
            self.res_block.add_module("residual" + str(1),
                                      BasicBlock(512, expansion=expansion))

        self.out_layer = nn.Sequential(nn.Linear(512, 512),
                                       nn.Linear(512, self.out_dim))

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block(x)
        x = self.out_layer(x)
        return x


# CNN MNIST
class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, 3),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, 3),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, 3),
                                   nn.ReLU())
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        #x = F.dropout(self.mpool(self.conv1(x)), p=0.2)
        #x = F.dropout(self.mpool(self.conv2(x)), p=0.3)
        #x = F.dropout(self.conv3(x), p=0.4)
        x = self.mpool(self.conv1(x))
        x = self.mpool(self.conv2(x))
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# LSTM + shakespeare
class RNN_Shakespeare(nn.Module):
    def __init__(self, vocab_size=80, embedding_dim=8, hidden_size=256):
        """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
        Args:
            vocab_size (int, optional): the size of the vocabulary, used as a dimension in the input embedding,
                Defaults to 80.
            embedding_dim (int, optional): the size of embedding vector size, used as a dimension in the output embedding,
                Defaults to 8.
            hidden_size (int, optional): the size of hidden layer. Defaults to 256.
        Returns:
            A `torch.nn.Module`.
        Examples:
            RNN_Shakespeare(
              (embeddings): Embedding(80, 8, padding_idx=0)
              (lstm): LSTM(8, 256, num_layers=2, batch_first=True)
              (fc): Linear(in_features=256, out_features=90, bias=True)
            ), total 819920 parameters
        """
        super(RNN_Shakespeare, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output