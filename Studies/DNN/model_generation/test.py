import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sample_type_lookup import lookup

mplhep.style.use(mplhep.styles.CMS)

class Tester:

    def __init__(self, testing_data, testing_df, device=None):
        self.x_data, self.y_data = testing_data
        self.testing_df = testing_df
        self.device = device

    def test(self, model):
        outputs = np.zeros(len(self.x_data))
        model.eval()
        with torch.no_grad():
            print("Running testing...")
            total = len(self.x_data)
            for i, sample in tqdm(enumerate(self.x_data), total=total):
                if self.device is not None:
                    x = torch.tensor(sample, device=self.device)
                else:
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
        h1 = np.histogram(signal, range=(0, 1), bins=50)
        h2 = np.histogram(background, range=(0, 1), bins=50)
        mplhep.histplot([h1,h2], label=['Signal','Background'], stack=True)
        if log:
            plt.yscale("log")
        plt.xlabel("Network output")
        plt.ylabel("Events")
        plt.legend()
        if log:
            output_name = "model_hist_log.png"
        else:
            output_name = "model_hist_lin.png"
        if show:
            plt.show()
        else:
            plt.savefig(output_name, bbox_inches="tight")


    def make_stackplot(self, log=False, show=False):
        # Init plot
        plt.clf()
        df = self.testing_df
        # Add individual hist curves
        for p in sorted(pd.unique(df.sample_type)):
            selected = df[df.sample_type == p].NN_Output
            h = np.histogram(selected, range=(0,1), bins=50)
            plt.stairs(*h, label=lookup[p])
        # Plot config
        if log:
            plt.yscale("log")
        plt.xlabel("Network output")
        plt.ylabel("Events")
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig("stackplot.png", bbox_inches="tight")
