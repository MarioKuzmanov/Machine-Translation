from implementation.constants import *
import json
import pandas as pd


class Loader(object):
    def __init__(self):
        graphemes = {'д', 'з', 'щ', 'м', 'т', 'в', 'е', 'ъ', 'р',
                     'н', 'ю',
                     'ш', 'ц', 'а', 'с', 'о', 'г', 'п', 'ь', 'б',
                     'к', 'и', 'я', 'ж', 'л', 'ф', 'х', 'у', 'й', 'ч', 'ѝ', BOS, EOS}

        with open("data/bul.json", "rt", encoding="utf8") as f:
            d1 = json.load(f)
            phonemes = {'ɡ'}
            for k, v in d1.items():

                phonemes.add(k)
                for phone in v:
                    phonemes.add(phone)

        self.graphemes = graphemes
        self.phonemes = phonemes

        self.df_train, self.df_val, self.df_test = None, None, None

    def load_datasets(self):
        self.df_train = pd.read_csv("../data/bul_train.tsv", sep="\t",
                                    header=None)
        self.df_val = pd.read_csv("../data/bul_val.tsv", sep="\t", header=None)
        self.df_test = pd.read_csv("../data/bul_test.tsv", sep="\t",
                                   header=None)
