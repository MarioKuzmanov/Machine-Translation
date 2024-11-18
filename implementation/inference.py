import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from implementation.architectures import *
from implementation.loader import *
from implementation.trainer import *


class Inference(object):

    def __init__(self, path2model):
        self.checkpoint = torch.load(path2model, map_location="cpu")

        self.loader = Loader()

        print(
            f"Training\nBest Epoch: {self.checkpoint['Epoch']}\nBest WER: {self.checkpoint['Wer']}\n"
            f"Loss: {self.checkpoint['Loss']}")

    def predict(self, w):
        networkSource = NetworkSource(self.loader.graphemes, self.checkpoint['VectorSize'])
        networkSource.load_state_dict(self.checkpoint['StateSource'])

        networkTarget = NetworkTarget(self.loader.phonemes, self.checkpoint['VectorSize'])
        networkTarget.load_state_dict(self.checkpoint['StateTarget'])

        h, c = networkSource(w)
        y_hat = networkTarget(w, h, c, train=False, prettyPrint=True)[0]
        return y_hat
