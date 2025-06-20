import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

mplhep.style.use(mplhep.styles.CMS)


class Tester:

    def __init__(self, testing_data, testing_df, device=None):
        (self.x_data, self.y_data), _ = testing_data
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
        self.testing_df["NN_Output"] = outputs

    def make_hist(self, weight=False, log=False, show=False):
        """
        Saves a histo of all signal vs all background
        """
        plt.clf()
        results = self.testing_df
        signal = results[results.Label == 1]
        background = results[results.Label == 0]
        if weight:
            h1 = np.histogram(
                signal.NN_Output, range=(0, 1), bins=50, weights=signal.Class_Weight
            )
            h2 = np.histogram(
                background.NN_Output,
                range=(0, 1),
                bins=50,
                weights=background.Class_Weight,
            )
        else:
            h1 = np.histogram(signal.NN_Output, range=(0, 1), bins=50)
            h2 = np.histogram(background.NN_Output, range=(0, 1), bins=50)
        mplhep.histplot([h1, h2], label=["Signal", "Background"], stack=True)
        # Set plot parameters based on boolean options
        if log:
            plt.yscale("log")
            output_name = "model_hist_log"
        else:
            output_name = "model_hist_lin"
        if weight:
            plt.ylabel("Weight")
            output_name += "_weighted"
        else:
            plt.ylabel("Events")
        # Set rest of plot and save/show
        plt.xlabel("Network output")
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".png", bbox_inches="tight")

    def make_stackplot(self, weight=False, log=False, show=False):
        """
        Saves a histo with the different processes draw independently
        """
        # Init plot
        plt.clf()
        df = self.testing_df
        # Add individual hist curves
        for p in sorted(pd.unique(df.sample_name)):
            selected = df[df.sample_name == p]
            if weight:
                h = np.histogram(
                    selected.NN_Output,
                    weights=selected.Class_Weight,
                    range=(0, 1),
                    bins=50,
                )
            else:
                h = np.histogram(selected.NN_Output, range=(0, 1), bins=50)
            plt.stairs(*h, label=p)
        # Plot config from boolean parameters
        output_name = "stackplot"
        if log:
            plt.yscale("log")
        if weight:
            output_name += "_weighted"
            plt.ylabel("Weight")
        else:
            plt.ylabel("Events")
        # Finish plot config
        plt.xlabel("Network output")
        plt.legend()
        # Out (save or show)
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".png", bbox_inches="tight")
