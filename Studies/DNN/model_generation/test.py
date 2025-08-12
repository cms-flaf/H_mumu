import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

mplhep.style.use(mplhep.styles.CMS)


class Tester:


    def __init__(
        self,
        testing_data,
        testing_df,
        signal_types,
        classification,
        n_bins=30,
        device=None,
        **kwargs,
    ):

        self.testing_df = testing_df
        self.device = device
        self.processes = sorted(pd.unique(testing_df.sample_name))
        self.signal_types = signal_types
        self.n_bins = n_bins
        self.hist_range = (0, 1)
        self.classification = classification

        (x_data, _), _ = testing_data
        self.x_data = torch.tensor(x_data, device=self.device)

        self.color_map = self._set_color_map()

    def _set_color_map(self):
        return {
            'DY' : 'gold',
            'TT' : 'blue',
            'VV' : 'lightslategray',
            'EWK' : 'magenta',
            'VBFH' : 'green',
            'ggH' : 'red' 
        }
        # processes = sorted(pd.unique(self.testing_df.sample_name))
        # colors = plt.color_sequences["tab10"]
        # return {p: c for p, c in zip(processes, colors)}

    def prediction_to_prob(self, subdf):
        classes = [x.replace("Prob_", "") for x in subdf.columns]
        predictions = subdf.apply(np.argmax, axis=1).apply(lambda x: classes[x])
        maxima = subdf.apply(np.max, axis=1).values.copy()
        mask = np.isin(predictions, self.signal_types)
        maxima[~mask] = 1 - maxima[~mask]
        return predictions, maxima

    def test(self, model):
        outputs = []
        model.eval()
        with torch.no_grad():
            print("Running testing...")
            total = len(self.x_data)
            # for i, sample in tqdm(enumerate(self.x_data), total=total):
            #     if self.device is not None:
            #         x = torch.tensor(sample, device=self.device)
            #     else:
            #         x = torch.Tensor(sample)
            #     guess = model.predict(x)
            #     if self.classification == 'binary':
            #         outputs.append(guess.item())
            #     else:
            #         outputs.append(guess.cpu().numpy())
            outputs = model.predict(self.x_data)
        if self.classification == "binary":
            self.testing_df["NN_Output"] = outputs.cpu().numpy()
        else:
            cols = [f"Prob_{x}" for x in self.processes]
            self.testing_df[cols] = outputs.cpu().numpy()
            predictions, probs = self.prediction_to_prob(self.testing_df[cols])
            self.testing_df["Prediction"] = predictions
            self.testing_df["NN_Output"] = probs

    ### Calculations of metrics and other ###
    
    def get_roc_auc(self):
        df = self.testing_df
        fpr, tpr, _ = roc_curve(df.Label, df.NN_Output, sample_weight=df.Class_Weight)
        score = np.trapz(1 - fpr, tpr)
        return score

    def by_hand_roc_calc(self):
        """
        Calculate signal efficiency and background rejection
        from the class histogram/stackplot parameters
        """
        df = self.testing_df
        sig = df[df.Label == 1]
        bkg = df[df.Label == 0]
        sig_hist, sig_bins = np.histogram(
            sig.NN_Output,
            weights=sig.Class_Weight,
            range=self.hist_range,
            bins=self.n_bins,
        )
        bkg_hist, bkg_bins = np.histogram(
            bkg.NN_Output,
            weights=bkg.Class_Weight,
            range=self.hist_range,
            bins=self.n_bins,
        )
        sig_eff = 1 - np.cumsum(sig_hist) / np.sum(sig_hist)
        bkg_rej = np.cumsum(bkg_hist) / np.sum(bkg_hist)
        return sig_eff, bkg_rej

    @staticmethod
    def s2overb(signal_bins, background_bins):
        """
        Takes two np arrays. Precompute a histogram and pass the bins over here.
        """
        x = (signal_bins**2)/background_bins
        return np.sqrt(np.sum(x**2))


    ### PLOTTING ###

    def make_hist(self, weight=True, log=False, norm=False, show=False):
        """
        Saves a histo of all signal vs all background
        """
        plt.clf()
        output_name = "model_hist"
        results = self.testing_df
        signal = results[results.Label == 1]
        background = results[results.Label == 0]
        if norm:
            w = sum(background.Class_Weight) / sum(signal.Class_Weight)
            output_name += "_normed"
        else:
            w = 1
        if weight:
            h1 = np.histogram(
                signal.NN_Output,
                range=self.hist_range,
                bins=self.n_bins,
                weights=signal.Class_Weight * w,
            )
            h2 = np.histogram(
                background.NN_Output,
                range=self.hist_range,
                bins=self.n_bins,
                weights=background.Class_Weight,
            )
        else:
            h1 = np.histogram(signal.NN_Output, range=self.hist_range, bins=self.n_bins)
            h2 = np.histogram(
                background.NN_Output, range=self.hist_range, bins=self.n_bins
            )
        plt.stairs(*h2, label="Background", color="tab:orange")
        plt.stairs(*h1, label="Signal", color="tab:blue")
        # Set plot parameters based on boolean options
        if log:
            plt.yscale("log")
            output_name += "_log"
        else:
            output_name += "_lin"
        if weight:
            plt.ylabel("Weight")
            output_name += "_weighted"
        else:
            plt.ylabel("Events")
        # Set rest of plot and save/show
        plt.xlabel("Network output")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        mplhep.cms.label(com=13.6)
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

    def make_multihist(self, weight=True, log=False, show=False):
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
                    range=self.hist_range,
                    bins=self.n_bins,
                )
            else:
                h = np.histogram(
                    selected.NN_Output, range=self.hist_range, bins=self.n_bins
                )
            plt.stairs(*h, label=p, color=self.color_map[p])
        # Plot config from boolean parameters
        output_name = "multihist"
        if log:
            plt.yscale("log")
        if weight:
            output_name += "_weighted"
            plt.ylabel("Weight")
        else:
            plt.ylabel("Events")
        # Finish plot config
        plt.xlabel("Network output")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        mplhep.cms.label(com=13.6)
        # Out (save or show)
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

    def make_stackplot(self, log=True, show=False):
        output_name = "stackplot"
        plt.clf()
        df = self.testing_df
        # Only stack background processes
        # Yeah yeah hardcoding sig is bad, I know.
        sig = ["VBFH", "ggH"]
        bkg = [x for x in pd.unique(df.sample_name) if x not in sig]

        # Make sure sorted for stacking
        def size(p):
            selected = df[df.sample_name == p]
            return selected.Class_Weight.sum()

        # Do the stack part plot
        bkg = sorted(bkg, key=lambda x: size(x))
        weights = []
        values = []
        for p in bkg:
            selected = df[df.sample_name == p]
            weights.append(selected.Class_Weight.values)
            values.append(selected.NN_Output.values)
        colors = [self.color_map[x] for x in bkg]
        plt.hist(
            values,
            weights=weights,
            bins=self.n_bins,
            range=self.hist_range,
            stacked=True,
            label=bkg,
            color=colors,
            alpha=0.9,
        )
        # Now add on the signal curves
        for p in sig:
            selected = df[df.sample_name == p]
            h = np.histogram(
                selected.NN_Output,
                weights=selected.Class_Weight,
                bins=self.n_bins,
                range=self.hist_range,
            )
            plt.stairs(*h, label=p, color=self.color_map[p])
        # Format
        if log:
            plt.yscale("log")
            output_name += "_log"
        plt.xlabel("Network output")
        plt.ylabel("Events")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        mplhep.cms.label(com=13.6)
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

    def make_roc_plot(self, hist_points=True, show=False):
        plt.clf()
        output_name = "roc"
        df = self.testing_df
        fpr, tpr, _ = roc_curve(df.Label, df.NN_Output, sample_weight=df.Class_Weight)
        plt.plot(tpr, 1 - fpr, label="sklearn")
        sig_eff, bkg_rej = self.by_hand_roc_calc()
        if hist_points:
            plt.scatter(sig_eff, bkg_rej, label="hists", color="tab:orange")
            output_name += "_whp"
        # Add 45 deg
        a = np.linspace(0, 1, 10)
        plt.plot(a, 1 - a, color="black", linestyle="dashed", label="unskilled")
        # Format and go!
        plt.xlabel("Sig. Efficiency")
        plt.ylabel("Bkg. Rejection")
        try:
            score = np.trapz(1 - fpr, tpr)
        except AttributeError:
            pass
        else:
            title = f"ROC, AUC = {round(score, 3)}"
            plt.title(title)
        # plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")



    def make_transformed_stackplot(self, log=True, show=False, min_power=-3, top_power=6):
        plt.clf()
        df = self.testing_df
        output_name = "transformed_stack"

        # Calculate bin edges for percentiles
        signal = df[df.Label == 1]
        q = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(signal.NN_Output, q)

        # In a specific order for AN_205 replication
        stack_proc = ["VV", "TT", "EWK", "DY"]
        other_proc = [x for x in pd.unique(df.sample_name) if x not in stack_proc]

        # Calculate the bin populations for each process
        counts_lookup = {}
        for p in pd.unique(df.sample_name):
            selected = df[df.sample_name == p]
            counts, _ = np.histogram(selected.NN_Output, weights=selected.Class_Weight, bins=bin_edges)
            counts_lookup[p] = counts

        # Do the stacking
        baseline = np.zeros(len(counts))
        total = np.zeros(len(baseline))
        x = np.linspace(0, 12, 13)
        for p in stack_proc:
            total += counts_lookup[p]
            plt.stairs(total, x, label=p, color=self.color_map[p], baseline=baseline, fill=True)
            baseline += counts_lookup[p]
        # and add the non-stackers
        for p in other_proc:
            plt.stairs(counts_lookup[p], x, label=p, color=self.color_map[p])

        # Calc s2/b for title
        sig = np.zeros(self.n_bins)
        for p in self.signal_types:
            sig += counts_lookup[p]
        score = self.s2overb(sig, total)
        text = r"$\frac{S^{2}}{B} =$"
        text += "{:.3e}".format(score)
        plt.text(1, 10**(top_power-1), text)
            
        # Format
        if log:
            output_name += "_log"
            plt.yscale("log")
        plt.xlabel(r"$DNN^{\prime}$")
        major_ticks = [1*10**x for x in range(min_power, top_power+1)]
        minor_ticks = [y*10**x for y in range(2,10) for x in range(min_power, top_power-1)]
        plt.ylim(10**min_power, 10**top_power)
        plt.ylabel("Events")
        plt.yticks(major_ticks)
        plt.yticks(minor_ticks, minor=True)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.legend()
        mplhep.cms.label(com=13.6)
        if show:
            plt.show()
        else:
            output_name += ".svg"
            plt.savefig(output_name, bbox_inches='tight')


