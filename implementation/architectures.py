import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from implementation.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetworkSource(nn.Module):

    def __init__(self, graphemes, vector_size):
        super(NetworkSource, self).__init__()

        self.graphemes = sorted(graphemes)
        self.grapheme2id = {self.graphemes[i]: i for i in range(len(self.graphemes))}

        self.vector_size = vector_size

        self.embedding_graphemes = nn.Embedding(len(self.grapheme2id), self.vector_size)

        self.encoder = nn.LSTM(input_size=self.vector_size, hidden_size=self.vector_size, num_layers=3,
                               bidirectional=True)

    def init_hidden(self):
        return torch.zeros(2 * 3, self.vector_size).to(device), torch.zeros(2 * 3, self.vector_size).to(device)

    def forward(self, ex):
        # tab-separated values
        graphic, _ = ex
        graphic = graphic.strip()

        graphic_tensor = torch.tensor(
            [self.grapheme2id[BOS]] + [self.grapheme2id[let] for let in graphic] + [self.grapheme2id[EOS]]).to(device)

        graphic_embedding = self.embedding_graphemes(graphic_tensor).to(device)

        h, c = self.init_hidden()
        _, (h, c) = self.encoder(graphic_embedding, (h, c))

        # sum the bidirectional layered representations
        h, c = torch.sum(h, dim=0).unsqueeze(0).to(device), torch.sum(c, dim=0).unsqueeze(0).to(device)

        return h, c


class NetworkTarget(nn.Module):

    def __init__(self, phonemes, vector_size):
        super(NetworkTarget, self).__init__()

        self.phonemes = ['</s>', '<s>'] + sorted(phonemes)
        self.phoneme2id = {self.phonemes[i]: i for i in range(len(self.phonemes))}
        self.id2phoneme = {i: self.phonemes[i] for i in range(len(self.phonemes))}

        self.vector_size = vector_size
        self.embedding_phonemes = nn.Embedding(len(self.phonemes), self.vector_size)

        self.decoder = nn.LSTM(input_size=self.vector_size, hidden_size=self.vector_size, num_layers=1,
                               bidirectional=False)

        self.layer = nn.Linear(in_features=self.vector_size, out_features=len(self.phonemes))

    def forward(self, ex, h, c, train=True, prettyPrint=True):
        # tab-separated values
        _, phonetic = ex
        phonetic = phonetic.strip().split(' ')

        phonetic_tensor = torch.tensor([self.phoneme2id[phone] for phone in phonetic] + [self.phoneme2id[EOS]]).to(
            device)

        # teacher forcing
        predictions, sample = [], torch.tensor(self.phoneme2id[BOS]).to(device)
        if train:
            idx = 0
            while idx < MAX_LENGTH and sample.item() != self.phoneme2id[EOS]:
                e = self.embedding_phonemes(sample).unsqueeze(0).to(device)
                o, (h, c) = self.decoder(e, (h, c))

                y_hat = self.layer(o)
                predictions.append(y_hat)

                sample = phonetic_tensor[idx]
                idx += 1
            predictions = torch.stack([p for p in predictions], dim=0).squeeze(1).to(device)
        else:
            with torch.no_grad():
                eos = 0
                while len(predictions) < MAX_LENGTH and sample.item() != eos:
                    e = self.embedding_phonemes(sample).unsqueeze(0).to(device)
                    o, (h, c) = self.decoder(e, (h, c))
                    y_hat = self.layer(o)
                    sample = torch.argmax(y_hat, dim=1).squeeze(0)

                    if prettyPrint:
                        predictions.append(self.id2phoneme[sample.item()])
                    else:
                        predictions.append(sample.item())
                if prettyPrint:
                    phonetic = [self.id2phoneme[id.item()] for id in phonetic_tensor]
                return " ".join(predictions[: -1]), " ".join(phonetic[: -1])

        return predictions, phonetic_tensor
