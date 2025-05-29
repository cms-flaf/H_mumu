import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Tester:

    def __init__(self, testing_data, testing_df):
        self.x_data, self.y_data = testing_data
        self.testing_df = testing_df

    def test(self, model):
        outputs = np.zeros(len(self.x_data))
        model.eval()
        with torch.no_grad():
            print("Running testing...")
            total = len(self.x_data)
            for i, sample in tqdm(enumerate(self.x_data), total=total):
                x = torch.Tensor(sample)
                guess = model.predict(x).item()
                outputs[i] = guess
        self.testing_df["Label"] = self.y_data
        self.testing_df["NN_Output"] = outputs

    def make_hist(self, log=False, show=False):
        """
        Saves a histo
        """
        plt.clf()
        results = self.testing_df
        signal = results[results.Label == 1].NN_Output
        background = results[results.Label == 0].NN_Output
        h1 = plt.hist(
            signal, range=(0, 1), bins=50, label="signal", color="tab:blue", alpha=0.6
        )
        h2 = plt.hist(
            background,
            range=(0, 1),
            bins=50,
            label="background",
            color="tab:orange",
            alpha=0.6,
        )
        if log:
            plt.yscale("log")
        plt.xlabel("NN output")
        plt.ylabel("Count (#)")
        plt.legend()
        if log:
            output_name = "model_hist_log.png"
        else:
            output_name = "model_hist_lin.png"
        if show:
            plt.show()
        else:
            plt.savefig(output_name, bbox_inches="tight")
