import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import ROOT
import matplotlib.ticker as ticker
import yaml
import re
import matplotlib.colors as mcolors
from HelpersForHistograms import *  # keep your helpers (resolve_text_positions etc.)
hep.style.use("CMS")



folder_names = {
 "inclusive_etainclusive" : "$p_T$ and $\eta$ incl",
 "inclusive_BB" : "$p_T$ incl, BB",
 "inclusive_BO" : "$p_T$ incl, BO",
 "inclusive_BE" : "$p_T$ incl, BE",
 "inclusive_OB" : "$p_T$ incl, OB",
 "inclusive_OO" : "$p_T$ incl, OO",
 "inclusive_OE" : "$p_T$ incl, OE",
 "inclusive_EB" : "$p_T$ incl, EB",
 "inclusive_EO" : "$p_T$ incl, EO",
 "inclusive_EE" : "$p_T$ incl, EE",
 "leading_mu_pt_upto26_etainclusive" : " $p_T$ > 26 GeV, $\eta$ incl",
 "leading_mu_pt_upto26_BB" : "$p_T$ > 26 GeV, BB",
 "leading_mu_pt_upto26_BO" : "$p_T$ > 26 GeV, BO",
 "leading_mu_pt_upto26_BE" : "$p_T$ > 26 GeV, BE",
 "leading_mu_pt_upto26_OB" : "$p_T$ > 26 GeV, OB",
 "leading_mu_pt_upto26_OO" : "$p_T$ > 26 GeV, OO",
 "leading_mu_pt_upto26_OE" : "$p_T$ > 26 GeV, OE",
 "leading_mu_pt_upto26_EB" : "$p_T$ > 26 GeV, EB",
 "leading_mu_pt_upto26_EO" : "$p_T$ > 26 GeV, EO",
 "leading_mu_pt_upto26_EE" : "$p_T$ > 26 GeV, EE",
 "leading_mu_pt_26to45_etainclusive" : "26 < $p_T$ < 45 GeV, $\eta$ incl",
 "leading_mu_pt_26to45_BB" : " 26 < $p_T$ < 45 GeV, BB",
 "leading_mu_pt_26to45_BO" : " 26 < $p_T$ < 45 GeV, BO",
 "leading_mu_pt_26to45_BE" : " 26 < $p_T$ < 45 GeV, BE",
 "leading_mu_pt_26to45_OB" : " 26 < $p_T$ < 45 GeV, OB",
 "leading_mu_pt_26to45_OO" : " 26 < $p_T$ < 45 GeV, OO",
 "leading_mu_pt_26to45_OE" : " 26 < $p_T$ < 45 GeV, OE",
 "leading_mu_pt_26to45_EB" : " 26 < $p_T$ < 45 GeV, EB",
 "leading_mu_pt_26to45_EO" : " 26 < $p_T$ < 45 GeV, EO",
 "leading_mu_pt_26to45_EE" : " 26 < $p_T$ < 45 GeV, EE",
 "leading_mu_pt_upto45_etainclusive" : " $p_T$ < 45 GeV,  $\eta$ incl",
 "leading_mu_pt_upto45_BB" : " $p_T$ < 45 GeV, BB",
 "leading_mu_pt_upto45_BO" : " $p_T$ < 45 GeV, BO",
 "leading_mu_pt_upto45_BE" : " $p_T$ < 45 GeV, BE",
 "leading_mu_pt_upto45_OB" : " $p_T$ < 45 GeV, OB",
 "leading_mu_pt_upto45_OO" : " $p_T$ < 45 GeV, OO",
 "leading_mu_pt_upto45_OE" : " $p_T$ < 45 GeV, OE",
 "leading_mu_pt_upto45_EB" : " $p_T$ < 45 GeV, EB",
 "leading_mu_pt_upto45_EO" : " $p_T$ < 45 GeV, EO",
 "leading_mu_pt_upto45_EE" : " $p_T$ < 45 GeV, EE",
 "leading_mu_pt_45to52_etainclusive" : " 45 < $p_T$ < 52 GeV, $\eta$ incl",
 "leading_mu_pt_45to52_BB" : " 45 < $p_T$ < 52 GeV, BB",
 "leading_mu_pt_45to52_BO" : " 45 < $p_T$ < 52 GeV, BO",
 "leading_mu_pt_45to52_BE" : " 45 < $p_T$ < 52 GeV, BE",
 "leading_mu_pt_45to52_OB" : " 45 < $p_T$ < 52 GeV, OB",
 "leading_mu_pt_45to52_OO" : " 45 < $p_T$ < 52 GeV, OO",
 "leading_mu_pt_45to52_OE" : " 45 < $p_T$ < 52 GeV, OE",
 "leading_mu_pt_45to52_EB" : " 45 < $p_T$ < 52 GeV, EB",
 "leading_mu_pt_45to52_EO" : " 45 < $p_T$ < 52 GeV, EO",
 "leading_mu_pt_45to52_EE" : " 45 < $p_T$ < 52 GeV, EE",
 "leading_mu_pt_52to62_etainclusive" : " 52 < $p_T$ < 62 GeV, $\eta$ incl",
 "leading_mu_pt_52to62_BB" : " 52 < $p_T$ < 62 GeV, BB",
 "leading_mu_pt_52to62_BO" : " 52 < $p_T$ < 62 GeV, BO",
 "leading_mu_pt_52to62_BE" : " 52 < $p_T$ < 62 GeV, BE",
 "leading_mu_pt_52to62_OB" : " 52 < $p_T$ < 62 GeV, OB",
 "leading_mu_pt_52to62_OO" : " 52 < $p_T$ < 62 GeV, OO",
 "leading_mu_pt_52to62_OE" : " 52 < $p_T$ < 62 GeV, OE",
 "leading_mu_pt_52to62_EB" : " 52 < $p_T$ < 62 GeV, EB",
 "leading_mu_pt_52to62_EO" : " 52 < $p_T$ < 62 GeV, EO",
 "leading_mu_pt_52to62_EE" : " 52 < $p_T$ < 62 GeV, EE",
 "leading_mu_pt_above62_etainclusive" : "$p_T$ > 62 GeV, $\eta$ incl",
 "leading_mu_pt_above62_BB" : " $p_T$ > 62 GeV, BB",
 "leading_mu_pt_above62_BO" : " $p_T$ > 62 GeV, BO",
 "leading_mu_pt_above62_BE" : " $p_T$ > 62 GeV, BE",
 "leading_mu_pt_above62_OB" : " $p_T$ > 62 GeV, OB",
 "leading_mu_pt_above62_OO" : " $p_T$ > 62 GeV, OO",
 "leading_mu_pt_above62_OE" : " $p_T$ > 62 GeV, OE",
 "leading_mu_pt_above62_EB" : " $p_T$ > 62 GeV, EB",
 "leading_mu_pt_above62_EO" : " $p_T$ > 62 GeV, EO",
 "leading_mu_pt_above62_EE" : " $p_T$ > 62 GeV, EE",
}

period_dict = {
    "Run3_2022": "7.9804",
    "Run3_2022EE": "26.6717",
    "Run3_2023": "18.063",
    "Run3_2023BPix": "9.693",
}

def compute_kde_from_hist(plot_vals, bin_edges):
    from scipy.stats import gaussian_kde
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    kde = gaussian_kde(x_centers, weights=plot_vals)
    x_dense = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y_dense = kde(x_dense)
    # integral_hist = np.sum(plot_vals)
    # integral_kde = np.trapz(y_dense, x_dense)

    # if integral_kde > 0:
    #     y_dense *= integral_hist / integral_kde
    return x_dense, y_dense

# -------------------------
# Utilities for hist arrays
# -------------------------
def get_bin_edges_widths(hist):
    nbins = hist.GetNbinsX()
    bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, nbins + 2)])
    bin_widths = np.array([hist.GetBinWidth(i) for i in range(1, nbins + 1)])
    return bin_edges, bin_widths


def get_hist_arrays(hist, divide_by_bin_width=False, scale=1.0):
    """
    Return: vals, errs, bin_edges, bin_widths
    vals, errs are numpy arrays length = nbins
    """
    bin_edges, bin_widths = get_bin_edges_widths(hist)
    nbins = hist.GetNbinsX()
    vals = np.array([hist.GetBinContent(i + 1) for i in range(nbins)], dtype=float) * scale
    errs = np.array([hist.GetBinError(i + 1) for i in range(nbins)], dtype=float) * scale
    if divide_by_bin_width:
        vals = np.divide(vals, bin_widths, out=np.zeros_like(vals), where=bin_widths != 0)
        errs = np.divide(errs, bin_widths, out=np.zeros_like(errs), where=bin_widths != 0)
    return vals, errs, bin_edges, bin_widths


def integral(hist, divide_by_bin_width=False):
    vals, _, _, bin_widths = get_hist_arrays(hist, divide_by_bin_width)
    if divide_by_bin_width:
        return float(np.sum(vals * bin_widths))
    return float(np.sum(vals))


def compute_total_mc_and_stat_err(mc_hists, divide_by_bin_width=False):
    """
    Sum MC per-bin and combine statistical errors in quadrature.
    mc_hists: dict name->TH1
    """
    if not mc_hists:
        return None, None
    first_hist = next(iter(mc_hists.values()))
    nbins = first_hist.GetNbinsX()
    total_vals = np.zeros(nbins, dtype=float)
    total_errs2 = np.zeros(nbins, dtype=float)
    for h in mc_hists.values():
        vals, errs, _, _ = get_hist_arrays(h, divide_by_bin_width)
        total_vals += vals
        total_errs2 += errs ** 2
    return total_vals, np.sqrt(total_errs2)


def choose_reference_binning(histograms_dict):
    """
    Return the first non-None histogram from a dict-like structure.
    """
    for name, h in histograms_dict.items():
        if h is None:
            continue
        return h
    return None


# -------------------------
# stack ordering
# -------------------------
def order_mc_contributions(mc_hists, divide_by_bin_width=False):
    """
    Default heuristic: reverse insertion order (same as before).
    Could be extended to read config stack_order.
    """
    names = list(mc_hists.keys())
    in_order = []
    remaining = [n for n in names if n not in in_order]
    remaining_reversed = list(reversed(remaining))
    return in_order + remaining_reversed


# -------------------------
# KDE utilities
# -------------------------
def _try_import_gaussian_kde():
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        return gaussian_kde
    except Exception:
        return None


def kde_from_binned(vals, bin_centers, bw, n_points=400, x_min=None, x_max=None):
    """
    Build a KDE-like smooth curve from a binned histogram:
      - vals: array of counts (or density) per bin center
      - bin_centers: x positions for bins
      - bw: gaussian kernel sigma (same units as x)
      - n_points: output resolution
    Returns xs, ys where ys have same integral (sum*binwidth) as input vals*binwidth.
    """
    if len(vals) == 0:
        return np.array([]), np.array([])

    if x_min is None:
        x_min = bin_centers[0] - 0.5 * (bin_centers[1] - bin_centers[0])
    if x_max is None:
        x_max = bin_centers[-1] + 0.5 * (bin_centers[-1] - bin_centers[-2] if len(bin_centers) > 1 else 0.0)

    xs = np.linspace(x_min, x_max, n_points)
    # gaussian kernel evaluated vectorized
    # construct (n_vals x n_xs) differences
    # use broadcasting: (xs[None, :] - centers[:, None])
    sigma = float(bw)
    if sigma <= 0:
        # fallback: no smoothing -> step interpolation
        ys = np.interp(xs, bin_centers, vals, left=0, right=0)
        return xs, ys

    # compute kernel contributions
    diffs = (xs[None, :] - bin_centers[:, None]) / sigma
    kernel = np.exp(-0.5 * diffs ** 2) / (sigma * np.sqrt(2 * np.pi))
    # weight by bin content
    ys = np.dot(vals, kernel)  # shape (n_xs,)
    # normalize: ensure integral(ys dx) equals sum(vals * binwidth)
    # compute input area:
    # approximate binwidth from centers spacing:
    if len(bin_centers) > 1:
        bin_width = np.diff(np.concatenate(([bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2],
                                           0.5 * (bin_centers[1:] + bin_centers[:-1]),
                                           [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2])))[0]
        # simpler: use mean spacing
        mean_binw = np.mean(np.diff(bin_centers))
    else:
        mean_binw = 1.0
    input_area = np.sum(vals * mean_binw)
    out_area = np.trapz(ys, xs)
    if out_area > 0:
        ys *= (input_area / out_area)
    return xs, ys


def compute_kde_for_hist(hist, divide_by_bin_width=False, bw=None, n_points=500):
    """
    Compute KDE-like smooth curve for a ROOT TH1 (no raw values required).
    bw: If None -> heuristic = mean bin width * 1.0 (adjustable)
    """
    vals, _, bin_edges, bin_widths = get_hist_arrays(hist, divide_by_bin_width)
    if len(vals) == 0:
        return np.array([]), np.array([])

    # bin centers
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # choose bw if not provided: typical choice ~ 1 * mean bin width
    if bw is None:
        bw = np.mean(bin_widths) * 1.0
    return kde_from_binned(vals, centers, bw=bw, n_points=n_points,
                           x_min=bin_edges[0], x_max=bin_edges[-1])


# -----------------------------
# Draw helpers: MC stack, signals, data
# -----------------------------
def draw_mc_stack(ax, mc_hists, processes_dict, bin_edges, divide_by_bin_width, page_cfg_dict):
    """
    Draw stacked MC and return (total_vals, total_errs)
    """
    if not mc_hists:
        return None, None
    order = order_mc_contributions(mc_hists, divide_by_bin_width)
    mc_vals, mc_labels, mc_colors = [], [], []

    for name in order:
        h = mc_hists[name]
        vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width)
        mc_vals.append(vals)
        cfg = processes_dict.get(name, {})
        mc_labels.append(cfg.get("name", name))
        mc_colors.append(cfg.get("color_mplhep", "gray"))

    total_mc_vals, total_mc_errs = compute_total_mc_and_stat_err(mc_hists, divide_by_bin_width)

    hep.histplot(
        mc_vals, bins=bin_edges, stack=True, histtype="fill",
        label=mc_labels, facecolor=mc_colors, edgecolor="black", linewidth=0.5, ax=ax
    )

    hep.histplot(
        total_mc_vals, bins=bin_edges, histtype="step",
        color="black", linewidth=0.5, ax=ax
    )

    bkg_unc_cfg = page_cfg_dict.get('bkg_unc_hist', {})
    unc_hatch = '//' if bkg_unc_cfg.get('fill_style') == 3013 else None
    unc_alpha = bkg_unc_cfg.get('alpha', 0.35)

    y_up = total_mc_vals + total_mc_errs
    y_dn = total_mc_vals - total_mc_errs
    y_dn = np.maximum(y_dn, 0.0)

    ax.fill_between(
        bin_edges[:-1], y_dn, y_up, step="post",
        facecolor="none", edgecolor="black", hatch=unc_hatch, alpha=unc_alpha,
        linewidth=0.8, label=bkg_unc_cfg.get('legend_title', 'Bkg. unc.')
    )

    return total_mc_vals, total_mc_errs


def draw_signals(ax, signal_hists, processes_dict, bin_edges, divide_by_bin_width, wantSignal):
    if not wantSignal or not signal_hists:
        return
    for name, h in signal_hists.items():
        cfg = processes_dict.get(name, {})
        scale = cfg.get("scale", 1.0)
        vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width, scale)
        label = cfg.get("name", name)
        if scale != 1.:
            label += f"x{scale}"
        hep.histplot(
            vals, bins=bin_edges, histtype="step",
            label=label,
            color=cfg.get("color_mplhep", "red"),
            linestyle="--", linewidth=1.5, ax=ax
        )


def draw_data(ax, data_hist, bin_edges, divide_by_bin_width, wantData=True, blind_region=None):
    if not wantData or data_hist is None:
        return None, None
    vals, errs, _, _ = get_hist_arrays(data_hist, divide_by_bin_width)

    # --- BLIND REGION ---
    if blind_region:
        if len(blind_region) == 2:
            x_min = blind_region[0]
            x_max = blind_region[1]
            mask = (bin_edges[:-1] >= x_min) & (bin_edges[:-1] < x_max)
            vals[mask] = 0.0
            errs[mask] = 0.0

    hep.histplot(vals, bins=bin_edges, yerr=errs, histtype="errorbar",
                 label="Data", color="black", ax=ax)
    return vals, errs


# -----------------------------
# Ratio
# -----------------------------
def draw_ratio(ax_ratio, bin_edges, data_vals, data_errs,
               total_mc_vals, total_mc_errs, x_label, blind_region):
    if data_vals is None or total_mc_vals is None:
        return

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(data_vals, total_mc_vals,
                          out=np.zeros_like(data_vals), where=total_mc_vals != 0)
        ratio_err = np.abs(np.array(np.divide(data_errs, total_mc_vals,
                                              out=np.zeros_like(data_errs), where=total_mc_vals != 0)))
        mc_rel_unc = np.divide(total_mc_errs, total_mc_vals,
                               out=np.zeros_like(total_mc_errs), where=total_mc_vals != 0)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # MC band
    y_up = 1.0 + mc_rel_unc
    y_dn = np.maximum(1.0 - mc_rel_unc, 0.0)

    mask = np.ones_like(ratio, dtype=bool)
    if blind_region and len(blind_region) == 2:
        x_min, x_max = blind_region
        mask = ~((bin_centers >= x_min) & (bin_centers <= x_max))
        blind_mask = ~mask
        ratio[blind_mask] = 0.0
        y_dn[blind_mask] = 0.0
        y_up[blind_mask] = 0.0

    ax_ratio.fill_between(bin_centers, y_dn, y_up, where=y_dn > 0,
                          step="mid", facecolor="ghostwhite",
                          edgecolor="black", hatch='//', alpha=0.5, zorder=1)

    ax_ratio.errorbar(bin_centers[mask], ratio[mask], yerr=ratio_err[mask], fmt='.', color='black', markersize=10, zorder=2)

    ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    delta = 0.5  # fallback
    if len(ratio[mask]):
        delta = np.abs(ratio[mask] - 1).mean()

    y_max = round(1 + delta, 2)
    y_min = round(1 - delta, 2)
    ax_ratio.set_ylim(y_min * 0.9, y_max * 1.1)
    ax_ratio.set_ylabel("Data/MC")
    ax_ratio.set_xlabel(x_label)


# -----------------------
# Main plotting function
# -----------------------
def plot_histogram_from_config(
    variable,
    histograms_dict,
    phys_model_dict,
    processes_dict,
    axes_cfg_dict,
    page_cfg_dict,
    page_cfg_custom_dict,
    filename_base,
    period,
    stacked=True,
    compare_mode=False,
    compare_vars_mode=False,
    wantLogX=False,
    wantLogY=False,
    wantData=False,
    wantSignal=False,
    wantRatio=False,
    category=None,
    channel=None,
    group_minor_contributions=False,
    minor_fraction=0.001
):
    """
    Modern refactor of the original plotting function with KDE support.
    - KDE configurable through page_cfg_dict["plot_options"] or axes_cfg_dict[variable].
      plot_options keys:
        enable_kde: bool
        kde_scope: "total_mc"|"components"|"data"|"signals"|"all"
        kde_bw: numeric (sigma in x units)
        kde_points: int
    """
    hist_cfg = axes_cfg_dict.get(variable, {})
    blind_region = hist_cfg.get("blind_region", [])
    divide_by_bin_width = bool(hist_cfg.get("divide_by_bin_width", False))

    # plot options (page-level override, then var-level)
    plot_opts = dict(page_cfg_dict.get("plot_options", {}))
    # per-variabile override
    var_plot_opts = hist_cfg.get("plot_options", {})
    plot_opts.update(var_plot_opts)

    enable_kde = bool(plot_opts.get("enable_kde", False))
    kde_scope = plot_opts.get("kde_scope", "total_mc")  # default
    kde_bw = plot_opts.get("kde_bw", None)
    kde_points = int(plot_opts.get("kde_points", 500))

    # Setup canvas and ratio
    canvas_size = page_cfg_dict['page_setup'].get('canvas_size', [1000, 800])
    ratio_plot = bool(wantData and wantRatio and stacked and not compare_mode)

    fig = plt.figure(figsize=(canvas_size[0] / 100, canvas_size[1] / 100))
    gs = fig.add_gridspec(
        2 if ratio_plot else 1, 1,
        height_ratios=[3, 1] if ratio_plot else [2],
        hspace=0.05 if ratio_plot else 0.25
    )
    ax = fig.add_subplot(gs[0])
    fig.subplots_adjust(top=0.85)
    ax_ratio = fig.add_subplot(gs[1], sharex=ax) if ratio_plot else None

    mc_hists = {}
    signal_hists = {}
    data_hist = None
    data_vals = data_errs = total_mc_vals = total_mc_errs = None
    y_max_comp = None

    # -------------------------
    # Compare mode (overlay regions)
    # -------------------------
    if compare_mode:
        linestyle_cycle = ['-', '--', ':', '-.']
        color_cycle = ['cornflowerblue', 'black', 'red', 'orange', 'gray', 'green', 'cyan', 'blue', 'magenta', 'purple']

        regions = list(histograms_dict.keys())
        first_region = next(iter(histograms_dict.values()))
        ref_hist = choose_reference_binning(first_region)
        if ref_hist is None:
            print("[plot_histogram_from_config] Nessun istogramma valido per il binning in compare_mode.")
            return

        _, _, bin_edges, _ = get_hist_arrays(ref_hist, False)
        all_plotted_vals = []

        for i, region in enumerate(regions):
            region_name = region
            if region in folder_names.keys():
                region_name = folder_names[region]
            hist_dict = histograms_dict[region]
            mc_hists_region = {k: h for k, h in hist_dict.items() if k in phys_model_dict.get('backgrounds', [])}
            plot_vals = None
            plot_label = ""

            if mc_hists_region:
                total_mc_vals_region, _ = compute_total_mc_and_stat_err(mc_hists_region, divide_by_bin_width)
                plot_vals = total_mc_vals_region
                plot_label = f"Total MC: {region_name}"
            elif hist_dict.get("data") is not None and wantData:
                plot_vals, _, _, _ = get_hist_arrays(hist_dict["data"], divide_by_bin_width)
                plot_label = f"Data: {region_name}"
            elif any(k in phys_model_dict.get('signals', []) for k in hist_dict) or  any(k in phys_model_dict.get('backgrounds', []) for k in hist_dict):
                s_name = next(k for k in hist_dict.keys())
                h = hist_dict[s_name]
                scale = processes_dict.get(s_name, {}).get("scale", 1.0)
                plot_vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width, scale)
                s_label = processes_dict.get(s_name, {}).get('name', s_name)
                if scale != 1.0:
                    s_label += f"x{scale}"
                plot_label = f"{s_label}: {region_name}"


            if plot_vals is None:
                continue

            all_plotted_vals.append(plot_vals)
            # If KDE enabled and we have valid plot_vals → compute KDE
            kde_x = None
            kde_y = None
            if plot_vals is not None and enable_kde:
                try:
                    kde_x, kde_y = compute_kde_from_hist(plot_vals, bin_edges)

                    max_hist = np.max(plot_vals)
                    max_kde = np.max(kde_y)

                    if max_kde > 0:
                        kde_y *= max_hist / max_kde

                    # print(kde_y)
                except Exception as exc:
                    print(f"[KDE] Warning: KDE failed in region {region}: {exc}")


            linestyle = linestyle_cycle[i % len(linestyle_cycle)]
            color = color_cycle[i % len(color_cycle)]
            if not enable_kde:
                hep.histplot(
                    plot_vals, bins=bin_edges, histtype="step",
                    color=color, linewidth=2, label=plot_label, ax=ax
                )
            else:
                if kde_x is not None and kde_y is not None:
                    # --- SOLO KDE (no overlay) ---
                    ax.plot(
                        kde_x, kde_y,
                        color=color,
                        # linestyle=linestyle,
                        linewidth=2,
                        label=f"{plot_label}",
                    )
                else:
                    # --- normale istogramma ---
                    hep.histplot(
                        plot_vals, bins=bin_edges,
                        histtype="step",
                        color=color,
                        linewidth=2,
                        label=plot_label,
                        ax=ax
                    )



        # if all_plotted_vals:
        #     max_vals = [np.max(v[np.isfinite(v)]) for v in all_plotted_vals if v.size > 0]
        #     y_max_comp = np.max(max_vals) if max_vals else 0

    # -------------------------
    # Compare variables mode
    # -------------------------
    elif compare_vars_mode:
        linestyle_cycle = ['-', '--', ':', '-.', ':', '-']
        color_cycle = ['blue', 'green', 'red', 'cyan']
        var_styles = {
            var: (linestyle_cycle[i % len(linestyle_cycle)], color_cycle[i % len(color_cycle)])
            for i, var in enumerate(histograms_dict.keys())
        }
        first_hist = next(iter(histograms_dict.values()))
        _, _, bin_edges, _ = get_hist_arrays(first_hist, False)
        linewidth = len(color_cycle) / 2
        alpha = 0.5
        all_plotted_vals = []

        for var, total_hist in histograms_dict.items():
            if total_hist is None:
                continue
            values = np.array([total_hist.GetBinContent(i + 1) for i in range(total_hist.GetNbinsX())])
            all_plotted_vals.append(values)
            style = var_styles.get(var)
            hep.histplot(
                values, bins=bin_edges, histtype="step",
                color=style[1], linewidth=linewidth, label=var, ax=ax, alpha=alpha,
            )
            linewidth -= 0.2 if linewidth > 0 else 0.1
            alpha += 0.25 / len(color_cycle)

        if all_plotted_vals:
            max_vals = [np.max(v[np.isfinite(v)]) for v in all_plotted_vals if v.size > 0]
            y_max_comp = np.max(max_vals) if max_vals else 0

    # -------------------------
    # Stack / standard mode
    # -------------------------
    else:
        for contrib, hist in histograms_dict.items():
            if hist is None:
                continue
            if contrib == "data":
                data_hist = hist
            elif contrib in phys_model_dict.get('signals', []):
                signal_hists[contrib] = hist
            elif contrib in phys_model_dict.get('backgrounds', []):
                mc_hists[contrib] = hist
            else:
                print(f"ref not found for {contrib}")

        # reference binning
        ref_hist = None
        if mc_hists:
            ref_hist = next(iter(mc_hists.values()))
        elif data_hist is not None:
            ref_hist = data_hist
        elif signal_hists:
            ref_hist = next(iter(signal_hists.values()))
        else:
            ref_hist = choose_reference_binning(histograms_dict)

        if ref_hist is None:
            print("[plot_histogram_from_config] Nessun istogramma valido trovato.")
            return

        _, _, bin_edges, _ = get_hist_arrays(ref_hist, False)

        total_mc_vals = total_mc_errs = None

        # group minor contributions if requested
        mc_hists_withMinor = mc_hists.copy()
        mc_order_withMinor = []

        if group_minor_contributions and mc_hists:
            integrals = {c: mc_hists[c].Integral() for c in mc_hists}
            total = sum(integrals.values())
            threshold = minor_fraction * total
            minor_contribs = [c for c, val in integrals.items() if val < threshold]
            major_contribs = [c for c in mc_hists.keys() if c not in minor_contribs]

            if minor_contribs:
                objsToMerge = ROOT.TList()
                other_hist = mc_hists[minor_contribs[0]].Clone(f"Other_{ref_hist.GetName()}")
                for minor_contrib in minor_contribs[1:]:
                    objsToMerge.Add(mc_hists[minor_contrib])
                other_hist.Merge(objsToMerge)
                # build new dict
                mc_hists_withMinor = {c: mc_hists[c] for c in mc_hists if c not in minor_contribs}
                mc_hists_withMinor["Other"] = other_hist
                mc_order_withMinor = ["Other"] + major_contribs

        # draw MC
        if mc_hists_withMinor and stacked:
            total_mc_vals, total_mc_errs = draw_mc_stack(
                ax, mc_hists_withMinor, processes_dict, bin_edges, divide_by_bin_width, page_cfg_dict
            )
        elif mc_hists_withMinor and not stacked:
            total_mc_vals, total_mc_errs = compute_total_mc_and_stat_err(mc_hists_withMinor, divide_by_bin_width)
            hep.histplot(total_mc_vals, bins=bin_edges, histtype="fill", facecolor="gray", alpha=0.35, ax=ax)
            for name, h in mc_hists_withMinor.items():
                vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width)
                cfg = processes_dict.get(name, {})
                hep.histplot(vals, bins=bin_edges, histtype="step",
                             label=cfg.get("title", name),
                             color=cfg.get("color_mplhep", "black"), linewidth=2, ax=ax)

        # signals and data
        draw_signals(ax, signal_hists, processes_dict, bin_edges, divide_by_bin_width, wantSignal)

        if data_hist is not None and wantData:
            data_vals, data_errs = draw_data(ax, data_hist, bin_edges, divide_by_bin_width, wantData, blind_region)

    # -------------------------
    # KDE overlay (flexible)
    # -------------------------
    # kde_scope: where to apply kde: "total_mc", "components", "data", "signals", "all"
    if enable_kde:
        # helper to plot kde for a single hist (ROOT.TH1)
        def _plot_kde_for_hist(name, hist, label_prefix=None, color=None, linestyle='-'):
            xs, ys = compute_kde_for_hist(hist, divide_by_bin_width=divide_by_bin_width, bw=kde_bw, n_points=kde_points)
            if xs.size == 0:
                return
            # choose color/label
            cfg = processes_dict.get(name, {})
            plot_color = color if color is not None else cfg.get("color_mplhep", None)
            label = (label_prefix or cfg.get("name", name))
            # if bins are densities (divide_by_bin_width), ys already in density units - keep area match
            ax.plot(xs, ys, linewidth=1.6, linestyle=linestyle, label=f"{label} (KDE)", color=plot_color, alpha=0.9)

        # total MC KDE
        if kde_scope in ("total_mc", "all"):
            if mc_hists:
                # build pseudo-total histogram by summing
                total_hist_clone = None
                for i, (n, h) in enumerate(mc_hists.items()):
                    if i == 0:
                        total_hist_clone = h.Clone(f"__total_mc_{h.GetName()}")
                    else:
                        total_hist_clone.Add(h)
                if total_hist_clone is not None:
                    _plot_kde_for_hist("TotalMC", total_hist_clone, label_prefix="Total MC", color="black", linestyle='-')

        # components KDE
        if kde_scope in ("components", "all"):
            for name, h in mc_hists.items():
                _plot_kde_for_hist(name, h, color=processes_dict.get(name, {}).get("color_mplhep", None), linestyle='-')

        # signals KDE
        if kde_scope in ("signals", "all"):
            for name, h in signal_hists.items():
                _plot_kde_for_hist(name, h, color=processes_dict.get(name, {}).get("color_mplhep", None), linestyle='--')

        # data KDE
        if kde_scope in ("data", "all"):
            if data_hist is not None:
                _plot_kde_for_hist("data", data_hist, label_prefix="Data", color="black", linestyle=':')

    # -------------------------
    # Axes, scales, limits
    # -------------------------
    ax.set_ylabel(hist_cfg.get("y_title", "Events"), fontsize=14)
    if not ratio_plot:
        ax.set_xlabel(hist_cfg.get("x_title", variable), fontsize=14)
    else:
        ax.get_xaxis().set_visible(False)

    ax.set_yscale("log" if wantLogY else "linear")
    ax.set_xscale("log" if wantLogX else "linear")

    # Determine y_max
    y_max = None
    if compare_mode or compare_vars_mode:
        y_max = y_max_comp
    elif mc_hists_withMinor:
        if total_mc_vals is None:
            total_mc_vals, _ = compute_total_mc_and_stat_err(mc_hists_withMinor, divide_by_bin_width)
        y_max = np.max(total_mc_vals) if total_mc_vals is not None and len(total_mc_vals) else None
    elif signal_hists:
        first_signal = next(iter(signal_hists.values()))
        vals, _, _, _ = get_hist_arrays(first_signal, False)
        y_max = np.max(vals) if vals is not None and len(vals) else None

    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        max_factor = hist_cfg.get("max_y_sf", 1.2) if not wantLogY else (10 ** (hist_cfg.get("max_y_sf", 1.0)))
        ax.set_ylim(top=y_max * max_factor)
        if wantLogY:
            y_min_log = max(0.1, y_max * 1e-4)
            ax.set_ylim(bottom=y_min_log)

    # X limits
    try:
        ax.set_xlim(bin_edges[0] * 0.99, bin_edges[-1] * 1.01)
    except Exception:
        pass

    # Legend
    legend_cfg = page_cfg_dict.get("legend_mplhep", {})
    ax.legend(
        loc='upper right',
        facecolor=legend_cfg.get("fill_color_mplhep", "white"),
        frameon=bool(legend_cfg.get("border_size", 0) == 0),
        fontsize=legend_cfg.get("text_size", 0.02) * 60,
        framealpha=0.0,
        ncol=legend_cfg.get("ncols", 2),
        handleheight=1.5,
        labelspacing=0.5
    )

    # -------------------------
    # Ratio plot
    # -------------------------
    if ratio_plot:
        if data_vals is not None and total_mc_vals is not None:
            draw_ratio(ax_ratio, bin_edges, data_vals, data_errs, total_mc_vals, total_mc_errs,
                       x_label=hist_cfg.get("x_title", variable), blind_region=blind_region)

    # -------------------------
    # CMS label and custom texts
    # -------------------------
    text_box_names = page_cfg_dict["page_setup"].get("text_boxes_mplhep", [])
    text_box_cfg = {name: page_cfg_dict.get(name, {}) for name in text_box_names}
    try:
        resolved_positions = resolve_text_positions(text_box_cfg)
    except NameError:
        resolved_positions = {}

    for name in text_box_names:
        cfg = text_box_cfg.get(name, {})
        pos = resolved_positions.get(name, cfg.get("pos", [0.02, 1.05]))
        if cfg.get("type") == "cms_mplhep":
            hep.cms.label(
                label="Preliminary", data=("data" in histograms_dict), ax=ax, loc=0,
                com=cfg.get("com", "13.6 TeV"),
                lumi=cfg.get("lumi", period_dict.get(period, "")),
                year=period.split('_')[1] if '_' in period else "Unknown",
                fontsize=cfg.get("text_size", 12)
            )
        else:
            text_content = cfg.get("text", "")
            text_content = text_content.format(category=category, channel=channel, variable=variable)
            ax.text(
                pos[0], pos[1], text_content, transform=ax.transAxes,
                fontsize=cfg.get("text_size", 10), ha="left", va="top"
            )

    # -------------------------
    # Save
    # -------------------------
    plt.savefig(f"{filename_base}.pdf", bbox_inches="tight")
    print(f"Plot saved to {filename_base}.pdf")
    plt.savefig(f"{filename_base}.png", bbox_inches="tight")
    print(f"Plot saved to {filename_base}.png")
    plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# import mplhep as hep
# import ROOT
# import matplotlib.ticker as ticker
# import yaml
# import re
# import matplotlib.colors as mcolors
# from HelpersForHistograms import *

# hep.style.use("CMS")

# period_dict = {
#     "Run3_2022": "7.9804",
#     "Run3_2022EE": "26.6717",
#     "Run3_2023": "18.063",
#     "Run3_2023BPix": "9.693",
# }

# def get_bin_edges_widths(hist):
#     nbins = hist.GetNbinsX()
#     bin_edges = np.array([hist.GetBinLowEdge(i) for i in range(1, nbins + 2)])
#     bin_widths = np.array([hist.GetBinWidth(i) for i in range(1, nbins + 1)])
#     return bin_edges, bin_widths

# def get_hist_arrays(hist, divide_by_bin_width=False, scale=1.0):
#     bin_edges, bin_widths = get_bin_edges_widths(hist)
#     nbins = hist.GetNbinsX()
#     vals = np.array([hist.GetBinContent(i + 1) for i in range(nbins)], dtype=float) * scale
#     errs = np.array([hist.GetBinError(i + 1) for i in range(nbins)], dtype=float) * scale
#     if divide_by_bin_width:
#         vals = np.divide(vals, bin_widths, out=np.zeros_like(vals), where=bin_widths != 0)
#         errs = np.divide(errs, bin_widths, out=np.zeros_like(errs), where=bin_widths != 0)
#     return vals, errs, bin_edges, bin_widths


# def integral(hist, divide_by_bin_width=False):
#     vals, _, _, bin_widths = get_hist_arrays(hist, divide_by_bin_width)
#     if divide_by_bin_width:
#         return float(np.sum(vals * bin_widths))
#     return float(np.sum(vals))

# def compute_total_mc_and_stat_err(mc_hists, divide_by_bin_width=False): # should add the option for pre-fit unc integration .. maybe it's not so trivial..
#     if not mc_hists:
#         return None, None
#     first_hist = next(iter(mc_hists.values()))
#     nbins = first_hist.GetNbinsX()
#     total_vals = np.zeros(nbins, dtype=float)
#     total_errs2 = np.zeros(nbins, dtype=float)
#     for h in mc_hists.values():
#         vals, errs, _, _ = get_hist_arrays(h, divide_by_bin_width)
#         total_vals += vals
#         total_errs2 += errs**2
#     return total_vals, np.sqrt(total_errs2)

# def choose_reference_binning(histograms_dict): # need to chose the reference binning when multiple hists are overlapped
#     for name, h in histograms_dict.items():
#         if h is None:
#             continue
#         return h
#     return None

# # -------------------------
# # Order for stacked plots
# # -------------------------
# def order_mc_contributions(mc_hists, divide_by_bin_width=False):

#     names = list(mc_hists.keys())
#     # where to define stack order? need to check --> before it was in the order of the samples in the input.yaml file .. now we need to figure out

#     # stack_order_cfg = inputs_cfg.get("stack_order", []) if isinstance(inputs_cfg, dict) else []
#     # in_order = [n for n in stack_order_cfg if n in names]
#     # in_order[::-1]

#     in_order = []
#     remaining = [n for n in names if n not in in_order]
#     remaining_reversed = list(reversed(remaining))
#     # order remaining by integral
#     # remaining_sorted_by_integral = list(sorted(remaining, key=lambda n: integral(mc_hists[n], divide_by_bin_width)))
#     # remaining_sorted_by_integral_reversed = list(sorted(remaining, key=lambda n: integral(mc_hists[n], divide_by_bin_width)), reverse=True)
#     return in_order + remaining_reversed

# # -----------------------------
# # Draw: MC stack, signal, data
# # -----------------------------

# def draw_mc_stack(ax, mc_hists, processes_dict, bin_edges, divide_by_bin_width, page_cfg_dict):
#     if not mc_hists:
#         return None, None
#     order = order_mc_contributions(mc_hists, divide_by_bin_width)
#     mc_vals, mc_labels, mc_colors = [], [], []

#     for name in order:
#         h = mc_hists[name]
#         vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width)
#         mc_vals.append(vals)
#         cfg = processes_dict[name]
#         mc_labels.append(cfg.get("name", name))
#         mc_colors.append(cfg.get("color_mplhep", "gray"))

#     total_mc_vals, total_mc_errs = compute_total_mc_and_stat_err(mc_hists, divide_by_bin_width)

#     hep.histplot(
#         mc_vals, bins=bin_edges, stack=True, histtype="fill",
#         label=mc_labels, facecolor=mc_colors, edgecolor="black", linewidth=0.5, ax=ax
#     )

#     hep.histplot(
#         total_mc_vals, bins=bin_edges, histtype="step",
#         color="black", linewidth=0.5, ax=ax
#     )

#     bkg_unc_cfg = page_cfg_dict.get('bkg_unc_hist', {})
#     unc_hatch = '//' if bkg_unc_cfg.get('fill_style') == 3013 else None
#     unc_alpha = bkg_unc_cfg.get('alpha', 0.35)

#     y_up = total_mc_vals + total_mc_errs
#     y_dn = total_mc_vals - total_mc_errs
#     y_dn = np.maximum(y_dn, 0.0)

#     ax.fill_between(
#         bin_edges[:-1], y_dn, y_up, step="post",
#         facecolor="none", edgecolor="black", hatch=unc_hatch, alpha=unc_alpha,
#         linewidth=0.8, label=bkg_unc_cfg.get('legend_title', 'Bkg. unc.')
#     )

#     return total_mc_vals, total_mc_errs

# def draw_signals(ax, signal_hists, processes_dict, bin_edges, divide_by_bin_width, wantSignal):
#     if not wantSignal or not signal_hists:
#         return
#     for name, h in signal_hists.items():
#         cfg = processes_dict[name]
#         scale = cfg.get("scale", 1.0)
#         vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width, scale)
#         label=cfg.get("name", name)
#         if scale !=1.:
#             label+=f"x{scale}"
#         hep.histplot(
#             vals, bins=bin_edges, histtype="step",
#             label=label,
#             color=cfg.get("color_mplhep", "red"),
#             linestyle="--", linewidth=1.5, ax=ax
#         )

# def draw_data(ax, data_hist, bin_edges, divide_by_bin_width, wantData=True, blind_region=[]):
#     if not wantData or data_hist is None:
#         return None, None
#     vals, errs, _, _ = get_hist_arrays(data_hist, divide_by_bin_width)

#     # --- BLIND REGION ---
#     if blind_region:
#         if len(blind_region) == 2 :
#             x_min = blind_region[0]
#             x_max = blind_region[1]
#             mask = (bin_edges[:-1] >= x_min) & (bin_edges[:-1] < x_max)
#             vals[mask] = 0.0
#             errs[mask] = 0.0
#         # to be expanded

#     hep.histplot(vals, bins=bin_edges, yerr=errs, histtype="errorbar",
#                  label="Data", color="black", ax=ax)
#     return vals, errs


# def draw_ratio(ax_ratio, bin_edges, data_vals, data_errs,
#                total_mc_vals, total_mc_errs, x_label, blind_region):
#     if data_vals is None or total_mc_vals is None:
#         return

#     with np.errstate(divide='ignore', invalid='ignore'):
#         ratio = np.divide(data_vals, total_mc_vals,
#                           out=np.zeros_like(data_vals), where=total_mc_vals != 0)
#         ratio_err = np.abs(np.array(np.divide(data_errs, total_mc_vals,
#                               out=np.zeros_like(data_errs), where=total_mc_vals != 0)))
#         mc_rel_unc = np.divide(total_mc_errs, total_mc_vals,
#                                out=np.zeros_like(total_mc_errs), where=total_mc_vals != 0)

#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # MC band
#     y_up = 1.0 + mc_rel_unc
#     y_dn = np.maximum(1.0 - mc_rel_unc, 0.0)

#     mask = np.ones_like(ratio, dtype=bool)
#     if blind_region and len(blind_region) == 2:
#         x_min, x_max = blind_region
#         mask = ~((bin_centers >= x_min) & (bin_centers <= x_max))
#         blind_mask = ~mask
#         ratio[blind_mask]=0.0
#         y_dn[blind_mask] = 0.0
#         y_up[blind_mask] = 0.0

#     # Disegna banda MC
#     ax_ratio.fill_between(bin_centers, y_dn, y_up, where=y_dn>0,
#                           step="mid", facecolor="ghostwhite",
#                           edgecolor="black", hatch='//', alpha=0.5, zorder=1)

#     ax_ratio.errorbar(bin_centers[mask], ratio[mask],yerr=ratio_err[mask], fmt='.', color='black', markersize=10, zorder=2)

#     ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
#     delta = 0.5  # valore di fallback
#     if len(ratio[mask]):
#         delta = np.abs(ratio[mask] - 1).mean()

#     # Centra y_min e y_max attorno a 1.0
#     y_max = round(1 + delta, 2)
#     y_min = round(1 - delta, 2)
#     print(f"y_max = {y_max}")
#     print(f"y_min = {y_min}")
#     # ax_ratio.set_ylim(0.9, 1.1)
#     ax_ratio.set_ylim(y_min*0.9, y_max*1.1)
#     # ax_ratio.yaxis.set_ticks(np.arange(0.9, 1.1, 0.05))
#     ax_ratio.set_ylabel("Data/MC")
#     ax_ratio.set_xlabel(x_label)
#     # for item in ax_ratio.get_yticklabels():
#         # item.set_fontsize(10)
#     # ax_ratio.set_ylim(0.9,1.1)
#     ax_ratio.set_ylabel("Data/MC")
#     ax_ratio.set_xlabel(x_label)


# # -----------------------
# # Funzione principale
# # -----------------------

# def plot_histogram_from_config(
#     variable,
#     histograms_dict,
#     phys_model_dict,
#     processes_dict,
#     axes_cfg_dict,
#     page_cfg_dict,
#     page_cfg_custom_dict,
#     filename_base,
#     period,
#     stacked=True,
#     compare_mode=False,
#     compare_vars_mode=False,
#     wantLogX=False,
#     wantLogY=False,
#     wantData=False,
#     wantSignal=False,
#     wantRatio=False,
#     category=None,
#     channel=None,
#     group_minor_contributions=False,
#     # signal_scale=1.0,       # << fattore globale segnali
#     # scale_dict=None,         # << dict per segnali individuali
#     minor_fraction=0.001      # << percentuale sotto cui raggruppare in 'other'
# ):
#     """
#     Plot degli istogrammi basato sulla configurazione.
#     - histograms_dict:
#         * compare_mode=False: dict {contrib_name: TH1}
#         * compare_mode=True:  dict {region: {contrib_name: TH1}}
#     """
#     # Config per la variabile
#     hist_cfg = axes_cfg_dict.get(variable, {})
#     blind_region = hist_cfg.get("blind_region", [])
#     divide_by_bin_width = bool(hist_cfg.get("divide_by_bin_width", False))

#     # Setup canvas e ratio
#     canvas_size = page_cfg_dict['page_setup'].get('canvas_size', [1000, 800])
#     # Ratio plot non ha senso in compare mode se si confronta solo MC, quindi lo disabilitiamo per compare_mode
#     ratio_plot = bool(wantData and wantRatio and stacked and not compare_mode)

#     fig = plt.figure(figsize=(canvas_size[0] / 100, canvas_size[1] / 100))
#     gs = fig.add_gridspec(
#         2 if ratio_plot else 1, 1,
#         height_ratios=[3, 1] if ratio_plot else [2],
#         hspace=0.05 if ratio_plot else 0.25
#     )
#     ax = fig.add_subplot(gs[0])
#     fig.subplots_adjust(top=0.85)
#     ax_ratio = fig.add_subplot(gs[1], sharex=ax) if ratio_plot else None

#     mc_hists = {}  # per evitare riferimenti successivi
#     mc_hists_withMinor = {}
#     data_vals = data_errs = total_mc_vals = total_mc_errs = None
#     y_max_comp = None # Variabile per il massimo in modalità compare

#     # -------------------------
#     # Modalità "compare" (region overlay) - FIX
#     # -------------------------
#     if compare_mode:
#         linestyle_cycle = ['-', '--', ':', '-.']
#         # Ciclo di colori standard per distinguere le regioni
#         color_cycle = ['cornflowerblue', 'black', 'red', 'orange', 'gray', 'green', 'cyan', 'blue', 'magenta','purple']
#         # color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
#         # BB: black, BO red, BE: orange, OB: gray, OO: green, OE: cyan, EB: blue, EO: magenta, EE: purple

#         regions = list(histograms_dict.keys())
#         # print(regions)
#         # 1. Choose reference binning from the first valid histogram in the first region
#         first_region = next(iter(histograms_dict.values()))

#         ref_hist = choose_reference_binning(first_region)
#         if ref_hist is None:
#             print("[plot_histogram_from_config] Nessun istogramma valido per il binning in compare_mode.")
#             return

#         _, _, bin_edges, _ = get_hist_arrays(ref_hist, False)

#         all_plotted_vals = []

#         # 2. Iterate and plot Total MC/Data for each region
#         for i, region in enumerate(regions):
#             hist_dict = histograms_dict[region]


#             # Separate background contributions
#             mc_hists_region = {k: h for k, h in hist_dict.items() if k in phys_model_dict.get('backgrounds', [])}

#             plot_vals = None
#             plot_label = ""

#             # Priorità 1: Total MC
#             if mc_hists_region:
#                 total_mc_vals_region, _ = compute_total_mc_and_stat_err(mc_hists_region, divide_by_bin_width)
#                 plot_vals = total_mc_vals_region
#                 plot_label = f"Total MC: {region}"
#             # Priorità 2: Data (se richiesto)
#             elif hist_dict.get("data") is not None and wantData:
#                 plot_vals, _, _, _ = get_hist_arrays(hist_dict["data"], divide_by_bin_width)
#                 plot_label = f"Data: {region}"
#             # Priorità 3: Primo Signal
#             elif any(k in phys_model_dict.get('signals', []) for k in hist_dict):
#                 s_name = next(k for k in hist_dict.keys() if k in phys_model_dict.get('signals', []))
#                 h = hist_dict[s_name]
#                 scale = processes_dict.get(s_name, {}).get("scale", 1.0)
#                 plot_vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width, scale)
#                 # Utilizziamo il nome del processo se definito
#                 s_label = processes_dict.get(s_name, {}).get('name', s_name)
#                 if scale != 1.0:
#                     s_label += f"x{scale}"
#                 plot_label = f"{s_label}: {region}"

#             if plot_vals is None:
#                 continue # Nessun contributo da plottare in questa regione

#             all_plotted_vals.append(plot_vals)
#             # print(all_plotted_vals)

#             # Assign style
#             linestyle = linestyle_cycle[i % len(linestyle_cycle)]
#             color = color_cycle[i % len(color_cycle)]

#             # Plot the resulting distribution
#             hep.histplot(
#                 plot_vals, bins=bin_edges, histtype="step", color=color,
#                  linewidth=2, label=plot_label, ax=ax , # linestyle=linestyle,
#             )

#         # Calcola il massimo di tutti i valori plottati per i limiti Y
#         if all_plotted_vals:
#             max_vals = [np.max(v[np.isfinite(v)]) for v in all_plotted_vals if v.size > 0]
#             y_max_comp = np.max(max_vals) if max_vals else 0

#     elif compare_vars_mode:
#         linestyle_cycle = ['-', '--', ':', '-.', ':', '-']
#         color_cycle = ['blue', 'green', 'red', 'cyan']#['cornflowerblue','pink','orange','green','yellow']#'#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
#         var_styles = {
#             var: (linestyle_cycle[i % len(linestyle_cycle)], color_cycle[i % len(color_cycle)])
#             for i, var in enumerate(histograms_dict.keys())
#         }
#         # reference binning choice from first histogram
#         first_hist = next(iter(histograms_dict.values()))
#         # print(first_hist)
#         # first_hist = next(iter(first_region.values()))
#         _, _, bin_edges, _ = get_hist_arrays(first_hist, False)
#         linewidth = len(color_cycle)/2

#         alpha=0.5
#         all_plotted_vals = []

#         for var, total_hist in histograms_dict.items():
#             if total_hist is None:
#                 continue

#             values = np.array([total_hist.GetBinContent(i + 1) for i in range(total_hist.GetNbinsX())])
#             all_plotted_vals.append(values)
#             style = var_styles.get(var)
#             hep.histplot(
#                 values, bins=bin_edges, histtype="step",
#                  color=style[1], linewidth=linewidth, label=var, ax=ax,alpha=alpha, #  linestyle=style[0],
#             )
#             linewidth-=0.2 if linewidth > 0 else 0.1
#             alpha+=0.25/len(color_cycle)

#         # Calcola il massimo di tutti i valori plottati per i limiti Y
#         if all_plotted_vals:
#             max_vals = [np.max(v[np.isfinite(v)]) for v in all_plotted_vals if v.size > 0]
#             y_max_comp = np.max(max_vals) if max_vals else 0

#     # -------------------------
#     # Stack
#     # -------------------------
#     else:
#         mc_hists, signal_hists, data_hist = {}, {}, None
#         for contrib, hist in histograms_dict.items():
#             if hist is None:
#                 continue
#             if contrib == "data":
#                 data_hist = hist
#             elif contrib in phys_model_dict['signals']:
#                 signal_hists[contrib] = hist
#             elif contrib in phys_model_dict['backgrounds']:
#                 mc_hists[contrib] = hist
#             else:
#                 print(f"ref not found for {contrib}")

#         # ref binning
#         ref_hist = None
#         if mc_hists:
#             ref_hist = next(iter(mc_hists.values()))
#         elif data_hist is not None:
#             ref_hist = data_hist
#         elif signal_hists:
#             ref_hist = next(iter(signal_hists.values()))
#         else:
#             ref_hist = choose_reference_binning(histograms_dict)

#         if ref_hist is None:
#             print("[plot_histogram_from_config] Nessun istogramma valido trovato.")
#             return

#         _, _, bin_edges, _ = get_hist_arrays(ref_hist, False)

#         total_mc_vals = total_mc_errs = None

#         ### minor contributions ###
#         mc_hists_withMinor = mc_hists
#         mc_order_withMinor = []


#         if group_minor_contributions:
#             integrals = {c: mc_hists[c].Integral() for c in mc_hists}
#             total = sum(integrals.values())
#             threshold = minor_fraction * total
#             minor_contribs = [c for c, val in integrals.items() if val < threshold]

#             print(minor_contribs)
#             # print(mc_hists)
#             # config_other = extract_config_for_sample("Other", inputs_cfg)
#             # print(config_other)
#             # for minor_contrib in minor_contribs:
#             #     if minor_contrib not in config_other['types']:
#             #         minor_contribs.remove(minor_contrib)
#             major_contribs = [c for c in histograms_dict.keys() if c not in minor_contribs]
#             print(major_contribs)

#             # if not minor_contribs:
#             #     return mc_hists, contributions
#             objsToMerge = ROOT.TList()
#             other_hist = mc_hists[minor_contribs[0]]
#             for minor_contrib in minor_contribs[1:]:
#                 objsToMerge.Add(mc_hists[minor_contrib])
#             other_hist.Merge(objsToMerge)
#             # for c in minor_contribs:
#             #     other_hist.Add(mc_hists[c])
#             # new_mc_hists = {c: mc_hists[c] for c in mc_hists if c not in minor_contribs}
#             # new_mc_hists[other_name] = other_hist
#             # new_contributions = major_contribs + [other_name]
#             mc_hists_withMinor = {c: mc_hists[c] for c in mc_hists if c not in minor_contribs}
#             mc_hists_withMinor["Other"] = other_hist
#             mc_order_withMinor =  ["Other"] + major_contribs


#         # Disegna stack MC
#         if mc_hists_withMinor and stacked:
#             total_mc_vals, total_mc_errs = draw_mc_stack(
#                 ax, mc_hists_withMinor, processes_dict, bin_edges, divide_by_bin_width, page_cfg_dict
#             )
#         elif mc_hists_withMinor and not stacked:
#             # versione non-stacked: plotta somma MC come fill + singoli contorni
#             total_mc_vals, total_mc_errs = compute_total_mc_and_stat_err(mc_hists_withMinor, divide_by_bin_width)
#             hep.histplot(total_mc_vals, bins=bin_edges, histtype="fill", facecolor="gray", alpha=0.35, ax=ax)
#             for name, h in mc_hists_withMinor.items():
#                 vals, _, _, _ = get_hist_arrays(h, divide_by_bin_width)
#                 cfg = processes_dict[name]
#                 hep.histplot(vals, bins=bin_edges, histtype="step",
#                              label=cfg.get("title", name),
#                              color=cfg.get("color_mplhep", "black"), linewidth=2, ax=ax)

#         # Disegna signals
#         draw_signals(ax, signal_hists, processes_dict, bin_edges, divide_by_bin_width, wantSignal)

#         # Disegna data
#         data_vals = data_errs = None
#         if data_hist is not None and wantData:
#             data_vals, data_errs = draw_data(ax, data_hist, bin_edges, divide_by_bin_width, wantData, blind_region)

#     # -------------------------
#     # Assi, scale, limiti
#     # -------------------------
#     ax.set_ylabel(hist_cfg.get("y_title", "Events"), fontsize=28)
#     if not ratio_plot:
#         ax.set_xlabel(hist_cfg.get("x_title", variable), fontsize=28)
#     else:
#         ax.get_xaxis().set_visible(False)

#     ax.set_yscale("log" if wantLogY else "linear")
#     ax.set_xscale("log" if wantLogX else "linear")

#     # Limiti Y (logica unificata)
#     y_max = None
#     if compare_mode or compare_vars_mode:
#         y_max = y_max_comp
#     elif mc_hists_withMinor:
#         if total_mc_vals is None:
#             total_mc_vals, _ = compute_total_mc_and_stat_err(mc_hists_withMinor, divide_by_bin_width)
#         y_max = np.max(total_mc_vals) if total_mc_vals is not None and len(total_mc_vals) else None
#     elif signal_hists:
#         # Check first signal
#         first_signal = next(iter(signal_hists.values()))
#         vals, _, _, _ = get_hist_arrays(first_signal, False)
#         y_max = np.max(vals)  if vals is not None and len(vals) else None

#     if y_max is not None and np.isfinite(y_max) and y_max > 0:
#             # Fattore di scala. In log, un fattore 1 significa 10x (10^1). In lineare, è un moltiplicatore diretto.
#             max_factor = hist_cfg.get("max_y_sf", 1.2) if not wantLogY else (10**(hist_cfg.get("max_y_sf", 1.0)))
#             ax.set_ylim(top=y_max * max_factor)
#             if wantLogY:
#                 # Imposta un limite inferiore ragionevole per la scala logaritmica (evitando zero)
#                 y_min_log = max(0.1, y_max * 1e-4)
#                 ax.set_ylim(bottom=y_min_log)


#     # Limiti X

#     ax.set_xlim(bin_edges[0] * 0.99, bin_edges[-1] * 1.01)


#     # Legenda
#     legend_cfg = page_cfg_dict.get("legend_mplhep", {})
#     ax.legend(
#         loc='upper right',
#         facecolor=legend_cfg.get("fill_color_mplhep", "white"),
#         frameon=bool(legend_cfg.get("border_size", 0) == 0),
#         fontsize=legend_cfg.get("text_size", 0.02) * 60,
#         framealpha=0.0,
#         ncol=legend_cfg.get("ncols", 2),
#         handleheight=1.5,
#         labelspacing=0.5
#     )

#     # -------------------------
#     # Ratio plot (Data/MC)
#     # -------------------------
#     if ratio_plot: # ratio_plot è già impostato a False in compare_mode
#         if data_vals is not None and total_mc_vals is not None:
#             draw_ratio(ax_ratio, bin_edges, data_vals, data_errs, total_mc_vals, total_mc_errs, x_label=hist_cfg.get("x_title", variable), blind_region=blind_region)

#     # -------------------------
#     # Label CMS e testo custom
#     # -------------------------
#     text_box_names = page_cfg_dict["page_setup"].get("text_boxes_mplhep", [])
#     text_box_cfg = {name: page_cfg_dict.get(name, {}) for name in text_box_names}
#     # Assumiamo che resolve_text_positions sia definito in HelpersForHistograms
#     # Se non è definito, questa riga causerà un errore e andrà gestita.
#     try:
#         resolved_positions = resolve_text_positions(text_box_cfg)
#     except NameError:
#         print("Warning: resolve_text_positions not defined. Using default positions.")
#         resolved_positions = {}

#     for name in text_box_names:
#         cfg = text_box_cfg[name]
#         pos = resolved_positions.get(name, [0.02, 1.05])
#         if cfg.get("type") == "cms_mplhep":
#             hep.cms.label(
#                 label="Preliminary", data="data" in histograms_dict, ax=ax, loc=0,
#                 com=cfg.get("com", "13.6 TeV"),
#                 lumi=cfg.get("lumi", period_dict.get(period, "")),
#                 # lumi=round(cfg.get("lumi", period_dict.get(period, "")),1),
#                 year=period.split('_')[1] if '_' in period else "Unknown",
#                 fontsize=cfg.get("text_size", 16)
#             )
#         else:
#             text_content = cfg.get("text", "")
#             text_content = text_content.format(category=category, channel=channel, variable=variable)
#             ax.text(
#                 pos[0], pos[1], text_content, transform=ax.transAxes,
#                 fontsize=cfg.get("text_size", 14), ha="left", va="top"
#             )

#     # -------------------------
#     # Salvataggio
#     # -------------------------
#     plt.savefig(f"{filename_base}.pdf", bbox_inches="tight")
#     print(f"Plot saved to {filename_base}.pdf")
#     plt.savefig(f"{filename_base}.png", bbox_inches="tight")
#     print(f"Plot saved to {filename_base}.png")
#     plt.close()
