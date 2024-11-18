import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from implementation.architectures import *
from implementation.loader import *
import evaluate
from copy import deepcopy
from tqdm import tqdm


class Trainer(object):
    def __init__(self, embedding_vector_size):
        self.embedding_vector_size = embedding_vector_size
        self.loader = Loader()
        self.loader.load_datasets()
        self.networkSource, self.networkTarget = None, None

    def train(self, epochs=50):
        self.networkSource = NetworkSource(self.loader.graphemes, self.embedding_vector_size).to(device)
        self.networkTarget = NetworkTarget(self.loader.phonemes, self.embedding_vector_size).to(device)

        loss_fn = nn.CrossEntropyLoss()

        optimizer1 = torch.optim.AdamW(self.networkSource.parameters(), lr=0.002, betas=(0.9, 0.9), weight_decay=1e-3)
        optimizer2 = torch.optim.AdamW(self.networkTarget.parameters(), lr=0.002, betas=(0.9, 0.9), weight_decay=1e-3)

        training_data = list(zip(self.loader.df_train[0], self.loader.df_train[1]))

        wer_prev, bestencoder, bestdecoder, bestoptim1, bestoptim2, bestepoch, bestwer, bestloss = float(
            "inf"), None, None, None, None, -1, None, None
        no_improve, loss_history = 0, []

        for epoch in range(epochs):
            loss_epoch, count = 0, 0

            if (epoch + 1) % 5 == 0:
                wer = self.evaluation(self.loader.df_val, self.networkSource, self.networkTarget)
                print("Validation data\nStep: {} Wer: {} Current best Wer: {}".format(epoch + 1, wer, wer_prev))
                if wer_prev > wer:
                    wer_prev = wer
                    no_improve = 0

                    # get best model
                    bestepoch = epoch + 1
                    bestencoder = deepcopy(self.networkSource.state_dict())
                    bestdecoder = deepcopy(self.networkTarget.state_dict())
                    bestoptim1 = deepcopy(optimizer1.state_dict())
                    bestoptim2 = deepcopy(optimizer2.state_dict())
                    bestwer = wer
                    bestloss = loss_history[-1]

                else:
                    # possibly, helps with overfitting
                    no_improve += 1
                    if no_improve == 4:
                        print("Training finished, no improvement on validation dataset.")
                        break

            for ex in tqdm(training_data):
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                h, c = self.networkSource(ex)
                p, t = self.networkTarget(ex, h, c, train=True, prettyPrint=False)

                loss = loss_fn(p, t)
                loss.backward()

                optimizer1.step()
                optimizer2.step()

                loss_epoch += loss.item()
                count += 1

            loss_history.append(loss_epoch / count)
            print(f"Epoch: {epoch + 1} Loss: {loss_epoch / count}")

        torch.save({"StateSource": bestencoder, "StateTarget": bestdecoder, "OptimSource": bestoptim1,
                    "OptimTarget": bestoptim2, "VectorSize": self.embedding_vector_size,
                    "Epoch": bestepoch, "Wer": bestwer, "Loss": bestloss},
                   "../bestmodel/translator.pt")

    def evaluation(self, data, network1, network2):

        d = list(zip(data[0], data[1]))

        wer = evaluate.load("wer")
        predictions, references = [], []

        for e in d:
            h, c = network1(e)
            p, t = network2(e, h, c, train=False, prettyPrint=True)

            predictions.append(p), references.append(t)

        return wer.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    # train , save and use the model
    trainer = Trainer(embedding_vector_size=200)
    trainer.train(epochs=50)
    wer = trainer.evaluation(data=trainer.loader.df_test, network1=trainer.networkSource,
                             network2=trainer.networkTarget)
    print(f"Test WER: {wer * 100}")
