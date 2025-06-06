import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sample_type_lookup import lookup

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


    def make_stackplot(self, log=False, show=False):
        # Init plot
        plt.clf()
        fig, (ax_plot, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

	df = self.testing_df
        # Add individual hist curves
        for p in sorted(pd.unique(df.sample_type)):
            selected = df[df. == p]
            counts, bin_edges = np.histogram(selected.NN_Output, bins=50, range=(0,1))
            x = np.concatenate(([0], bin_edges))
            y = np.concatenate(([0], counts, [0]))
            name = lookup[p]
            ax_plot.plot(x, y, label=name, drawstyle='steps-post')
        # Plot config
        ax_plot.set_xlim((0,1))
        if log:
            ax_plot.set_yscale("log")
        else:
            ax_plot.set_ylim(bottom=0)
        ax_plot.set_xlabel("NN Output")
        ax_plot.set_ylabel("Count (#)")
        # Create legend subplot
        h, l = ax_plot.get_legend_handles_labels()
        ax_legend.legend(h, l, loc='center', frameon=False, fontsize='medium')
        ax_legend.axis('off')
        # Output
        plt.tight_layout()
        if show:
            plt.show()
        else:
            fig.savefig("stackplot.png", bbox_inches="tight")
