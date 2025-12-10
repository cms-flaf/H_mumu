
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
import ROOT
from ROOT import RooFit as RF
import yaml

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

from Analysis.plotting_tools.HelpersForHistograms import *

# Definizione della funzione C++ per la Double Crystal Ball
ROOT_DCB_CODE = """
#include "TMath.h"

Double_t DoubleSidedCB(Double_t x, Double_t mu, Double_t sigma, Double_t alpha1, Double_t n1, Double_t alpha2, Double_t n2)
{
    Double_t z = (x - mu) / sigma;
    if (z > -alpha1 && z < alpha2) {
        return TMath::Exp(-0.5 * z * z);
    } else if (z <= -alpha1) {
        Double_t A1 = TMath::Power(n1 / TMath::Abs(alpha1), n1) * TMath::Exp(-0.5 * alpha1 * alpha1);
        Double_t B1 = n1 / TMath::Abs(alpha1) - TMath::Abs(alpha1);
        return A1 * TMath::Power(B1 - z, -n1);
    } else { // z >= alpha2
        Double_t A2 = TMath::Power(n2 / TMath::Abs(alpha2), n2) * TMath::Exp(-0.5 * alpha2 * alpha2);
        Double_t B2 = n2 / TMath::Abs(alpha2) - TMath::Abs(alpha2);
        return A2 * TMath::Power(B2 + z, -n2);
    }
}
"""
ROOT.gROOT.ProcessLine(ROOT_DCB_CODE)


def print_histogram_output(histogram_output):
    counts, bin_edges = histogram_output
    print(np.array(counts, dtype=int))
    print(bin_edges[0])

def do_fit_and_plot(
    hist,
    out_file_name,
    var_name,
    var_x_title,
    year,
    fit_func="DoubleSidedCB",
    bg_func="Exponential",
    fit_range=(85, 95),
    rebin_bins=None,
    isMC=False,
    hist_signal=None,
    hist_background=None,
    wantLogY=False
):
    if not hist or hist.IsZombie():
        print("Errore: istogramma non valido.")
        return

    if rebin_bins is not None and len(rebin_bins) > 0:
        hist = RebinHisto(hist, rebin_bins, hist.GetName() + "_rebin", wantOverflow=False, verbose=False)
        if isMC and hist_signal and hist_background:
            hist_signal = RebinHisto(hist_signal, rebin_bins, hist_signal.GetName() + "_rebin", wantOverflow=False, verbose=False)
            hist_background = RebinHisto(hist_background, rebin_bins, hist_background.GetName() + "_rebin", wantOverflow=False, verbose=False)

    if hist.Integral(hist.FindBin(fit_range[0]), hist.FindBin(fit_range[1])) <= 0:
        print(f"Warning: No events found in the histogram for the specified fit range [{fit_range[0]}, {fit_range[1]}].")
        print("The fit cannot be performed. Exiting...")
        return

    w = ROOT.RooWorkspace("w", "workspace")


    m = ROOT.RooRealVar(var_name, f"hist_{var_name}", fit_range[0], fit_range[1])
    m.setRange("full_range", fit_range[0], fit_range[1])

    data = ROOT.RooDataHist("data", "data", ROOT.RooArgList(m), hist)

    hist_mean = hist.GetMean()
    hist_stddev = hist.GetStdDev()

    # physical_mean = ROOT.RooRealVar("physical_mean", "Mean", 91.1880, 89.,  92.)
    physical_mean = ROOT.RooRealVar("physical_mean", "Mean", 91.1880, 91.1880-1.,  91.1880+1.) # pdg value within 3 sigmas (91.1880±0.0020)
    # physical_mean = ROOT.RooRealVar("physical_mean", "Mean", 91.1880, 91.1880-0.0020,  91.1880+0.0020) # pdg value within 3 sigmas (91.1880±0.0020)
    # physical_mean.setConstant(ROOT.kTRUE)
    physical_sigma = ROOT.RooRealVar("physical_sigma", "Sigma", 2.4955, 2.4955-0.0069, 2.4955+0.0069) # pdg value within 3 sigmas (2.4955±0.0023)
    physical_GammaHalf_BW = ROOT.RooRealVar("BW_gammaHalf", "Gamma/2", 2.4955/2, (2.4955-0.0069)/2, (2.4955+0.0069)/2) # pdg value within 3 sigmas (2.4955±0.0023)
    # physical_sigma.setConstant(ROOT.kTRUE)


    res_mean = ROOT.RooRealVar("res_mean", "Res_Mean", 0.0, -2.0, 2.0)
    res_mean.setConstant(ROOT.kTRUE) ## DSCB mean fixed to 0 == s
    res_sigma = ROOT.RooRealVar("res_sigma", "Res_Sigma", hist_stddev, 0.1, 5.0)


    alpha1_DSCB = ROOT.RooRealVar("alpha1", "Alpha1", 1.5, 0.5, 5)
    n1_DSCB = ROOT.RooRealVar("n1", "N1", 2.0, 0, 10)
    alpha2_DSCB = ROOT.RooRealVar("alpha2", "Alpha2", 1.5, 0.5, 5)
    n2_DSCB = ROOT.RooRealVar("n2", "N2", 2.0, 0, 10)
    alpha_CB = ROOT.RooRealVar("alpha", "Alpha", 1.5, 0.5, 5)
    n_CB = ROOT.RooRealVar("n", "N", 2.0, 0, 10)

    if fit_func == "DoubleSidedCB":
        sig_pdf = ROOT.RooGenericPdf("sig_pdf", "DoubleSidedCB(@0, @1, @2, @3, @4, @5, @6)",
                                     ROOT.RooArgList(m, physical_mean, physical_sigma, alpha1_DSCB, n1_DSCB, alpha2_DSCB, n2_DSCB))
        sig_pdf_name = "DSCB"

    elif fit_func == "CrystalBall":
        sig_pdf = ROOT.RooCBShape("sig_pdf", "CrystalBall", m, physical_mean, physical_sigma, alpha_CB, n_CB)
        sig_pdf_name = "CB"

    elif fit_func == "Voigtian":
        sig_pdf = ROOT.RooVoigtian("sig_pdf", "Voigtian", m, physical_mean, physical_sigma, res_sigma)
        sig_pdf_name = "Voigtian"

    elif fit_func == "BreitWigner":
        sig_pdf = ROOT.RooBreitWigner("bw_pdf", "BreitWigner", m, physical_mean, BW_gammaHalf)
        sig_pdf_name = "BreitWigner"

    elif fit_func == "BW_conv_DCS":
        bw_pdf = ROOT.RooBreitWigner("bw_pdf", "BreitWigner", m, physical_mean, physical_GammaHalf_BW)
        dcs_pdf = ROOT.RooGenericPdf("dcs_pdf", "DoubleSidedCB(@0, @1, @2, @3, @4, @5, @6)",
                                     ROOT.RooArgList(m, res_mean, res_sigma, alpha1_DSCB, n1_DSCB, alpha2_DSCB, n2_DSCB))

        # 3. Convoluzione (usa bw_pdf come PDF e dcs_pdf come kernel)
        m.setBins(1000, "cache") # for FFT
        sig_pdf = ROOT.RooFFTConvPdf("sig_pdf", "BW_conv_DCS", m, bw_pdf, dcs_pdf, 2)
        sig_pdf_name = "BW_conv_DCS"

    else:
        print(f"Errore: funzione di fit '{fit_func}' non supportata. Uso DoubleSidedCB di default.")
        sig_pdf = ROOT.RooGenericPdf("sig_pdf", "DoubleSidedCB(@0, @1, @2, @3, @4, @5, @6)",
                                     ROOT.RooArgList(m, physical_mean, physical_sigma, alpha1_DSCB, n1_DSCB, alpha2_DSCB, n2_DSCB))
        sig_pdf_name = "DSCB"

    if bg_func == "Exponential":
        bg_alpha = ROOT.RooRealVar("bg_alpha", "Alpha", -0.1, -1.0, 0.0)
        bg_pdf = ROOT.RooExponential("bg_pdf", "Background", m, bg_alpha)
        bg_pdf_name = "Exponential"

    elif bg_func == "Erf_conv_Exp":
        # 1. Componente Esponenziale (fisica)
        bg_alpha_exp = ROOT.RooRealVar("bg_alpha_exp", "Alpha_Exp", -0.05, -1.0, 0.0)
        exp_pdf = ROOT.RooExponential("exp_pdf", "Exponential_Kernel", m, bg_alpha_exp)

        # 2. Componente Erf (regione di taglio o cutoff)
        # RooFit non ha una RooErf nativa, ma usa la forma Erf*Polynomial (RooBernstein/RooChebychev con Erf)
        # Per una Erf 'pura' convoluta ad una Exp, usiamo un'implementazione via GenericPdf o un kernel Gaussian/DCB
        # (che sono spesso usati come kernel per la Erf in contesti di convoluzione).
        # Implementiamo la "Erf-turn-on" * Esponenziale (la forma più comune per i taglie)
        # La forma Erf * Exp è implementata come RooGenericPdf:

        c0 = ROOT.RooRealVar("c0", "c0", 1.0)
        bg_alpha = ROOT.RooRealVar("bg_alpha", "Alpha", -0.05, -1.0, 0.0)
        m0 = ROOT.RooRealVar("m0", "M0_Erf", fit_range[0], fit_range[0] - 5, fit_range[0] + 5) # Punto di turn-on
        sigma_erf = ROOT.RooRealVar("sigma_erf", "Sigma_Erf", 1.0, 0.1, 5.0) # Sharpness del turn-on

        # Implementazione della forma RooGenericPdf: exp(-alpha*m) * (1 + erf((m - m0) / (sqrt(2)*sigma)))
        # Non è una convoluzione, ma una PDF che modella il background in modo flessibile.
        bg_pdf = ROOT.RooGenericPdf(
            "bg_pdf",
            "TMath::Exp(@1 * @0) * (1 + TMath::Erf((@0 - @2) / (@3 * TMath::Sqrt(2))))",
            ROOT.RooArgList(m, bg_alpha, m0, sigma_erf)
        )
        bg_pdf_name = "Erf_x_Exp"
        # NOTA: Per un'implementazione VERA di convoluzione, potresti dover definire una PDF Erf in C++ e usare RooFFTConvPdf
        # ma la forma Erf*Exp è quella comunemente intesa quando si chiede la "Erf-convoluta" per il bckg.

    else: # Fallback a Esponenziale
        bg_alpha = ROOT.RooRealVar("bg_alpha", "Alpha", -0.1, -1.0, 0.0)
        bg_pdf = ROOT.RooExponential("bg_pdf", "Background", m, bg_alpha)
        bg_pdf_name = "Exponential"

    # # Define background model (Exponential is common for Z->mumu)
    # bg_alpha = ROOT.RooRealVar("bg_alpha", "Alpha", -0.1, -1.0, 0.0)
    # bg_pdf = ROOT.RooExponential("bg_pdf", "Background", m, bg_alpha)

    # Define total model and event counts
    # initial_integral = hist.Integral(hist.FindBin(fit_range[0]), hist.FindBin(fit_range[1]))
    # # if no MC info on nsig and nbg you suppose that 90% is signal and 10% is bg
    # nsig = ROOT.RooRealVar("nsig", "Number of signal events", initial_integral * 0.9, 0, initial_integral * 1.5)
    # nbg = ROOT.RooRealVar("nbg", "Number of background events", initial_integral * 0.1, 0, initial_integral * 1.5)




    # if isMC:
    initial_signal_integral = hist_signal.Integral(hist_signal.FindBin(fit_range[0]), hist_signal.FindBin(fit_range[1]))
    initial_background_integral = hist_background.Integral(hist_background.FindBin(fit_range[0]), hist_background.FindBin(fit_range[1]))
    nsig = ROOT.RooRealVar("nsig", "Number of signal events", initial_signal_integral, 0, initial_signal_integral*1.5)
    nbg = ROOT.RooRealVar("nbg", "Number of background events", initial_background_integral * 0.1, 0, initial_background_integral)



    model = ROOT.RooAddPdf("model", "Signal and Background", ROOT.RooArgList(sig_pdf, bg_pdf), ROOT.RooArgList(nsig, nbg))

    print(f"Starting the fit with {sig_pdf_name} signal model...")
    # Esegui il fit. Usa il Minimizer Minuit per stabilità
    fit_result = model.fitTo(data, RF.Range("full_range"), RF.Save(), RF.Verbose(False), RF.Minimizer("Minuit2", "migrad"))
    # print(f"*********** Beginning of Printing Fit result *************\n\n")
    # fit_result.Print()

    # # print(f"\n\n *********** Covariance and correlation matrices *************\n\n")
    # # Print correlation, matrix
    # cor = fit_result.correlationMatrix()
    # cov = fit_result.covarianceMatrix()
    # print("correlation matrix")
    # cor.Print()
    # print("covariance matrix")
    # cov.Print()

    # print(f"\n\n *********** floating parameters values *************\n\n")

    fit_result.floatParsFinal().Print("s")

    # print(f"\n\n*********** End of Printing Fit result *************")


    print(f"*********** Beginning of Printing Chi square stuff *************\n")
    n_params = fit_result.floatParsFinal().getSize()
    print(f"n_params = {n_params}")


    # Crea il RooPlot frame sul RANGE COMPLETO per estrarre le curve estrapolate
    frame = m.frame(RF.Title("RooFit Plot"), RF.Range("full_range"))
    data.plotOn(frame, RF.Name("data_plot"))
    model.plotOn(frame, RF.LineColor(ROOT.kRed), RF.LineWidth(2), RF.Name("total_fit"))
    chi2_val_noparams = frame.chiSquare("total_fit", "data_plot")
    print(f"chi2 WITHOUT considering parameters = {chi2_val_noparams}")
    chi2_val_tot = frame.chiSquare("total_fit", "data_plot", n_params)
    print(f"chi2 considering parameters = {chi2_val_tot}")
    print(f"\n*********** End of Printing Chi square stuff *************")

    print(f"Starting the model plotting with {sig_pdf_name} signal model...")
    resid_fit_hist = frame.residHist()
    frame2 = m.frame(RF.Title("RooFit Residual"), RF.Range("full_range"))
    frame2.addPlotable(resid_fit_hist, "P")
    c2 = ROOT.TCanvas("residuals", "residuals", 1000, 800)
    frame2.Draw()
    c2.SaveAs(f"{out_file_name}_residuals.pdf")
    pull_fit_hist = frame.pullHist()
    frame3 = m.frame(RF.Title("RooFit Pulls"), RF.Range("full_range"))
    frame3.addPlotable(pull_fit_hist, "P")
    c3 = ROOT.TCanvas("pulls", "pulls", 1000, 800)
    frame3.Draw()
    c3.SaveAs(f"{out_file_name}_pulls.pdf")
    model.plotOn(frame, RF.Components(ROOT.RooArgSet(bg_pdf)), RF.LineColor(ROOT.kGreen), RF.LineStyle(ROOT.kDashed), RF.LineWidth(2), RF.Name("bkg_fit"))
    model.plotOn(frame, RF.Components(ROOT.RooArgSet(sig_pdf)), RF.LineColor(ROOT.kBlue), RF.LineStyle(ROOT.kDashed), RF.LineWidth(2), RF.Name("sig_fit"))


    # --- Plotting with Matplotlib and mplhep ---
    bins = np.array(hist.GetXaxis().GetXbins())
    y_values = np.array([hist.GetBinContent(i + 1) for i in range(hist.GetNbinsX())])
    y_errors = np.array([hist.GetBinError(i + 1) for i in range(hist.GetNbinsX())])

    x_fit = np.linspace(fit_range[0], fit_range[1], 1000)
    bin_width = hist.GetBinWidth(1) # Larghezza del bin è cruciale per la normalizzazione

    # Calcolo corretto della curva di fit
    # RooAddPdf.getVal() restituisce la DENSITA' di eventi (Eventi / unità di massa).
    # Per plottarla su un istogramma di CONTEGGI (Eventi), si moltiplica per la larghezza del bin.
    y_fit_total = []
    y_fit_signal_pdf = [] # PDF del solo segnale (normalizzata a 1)
    y_fit_background_pdf = [] # PDF del solo background (normalizzata a 1)

    for x in x_fit:
        m.setVal(x)
        y_fit_total.append(model.getVal(ROOT.RooArgSet(m)))
        y_fit_signal_pdf.append(sig_pdf.getVal(ROOT.RooArgSet(m)))
        y_fit_background_pdf.append(bg_pdf.getVal(ROOT.RooArgSet(m)))

    # Le componenti sono le PDF * Conteggi * larghezza_bin
    y_fit_total = (nsig.getVal() + nbg.getVal()) * np.array(y_fit_total) * bin_width
    y_fit_signal = nsig.getVal() * np.array(y_fit_signal_pdf) * bin_width
    y_fit_background = nbg.getVal() * np.array(y_fit_background_pdf) * bin_width


    # --- Inizio Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))



    # Plot delle curve di fit

    # Plot MC, se richiesto
    if isMC:
        if hist_signal:
            hep.histplot(
                np.array([hist_signal.GetBinContent(i + 1) for i in range(hist_signal.GetNbinsX())]),
                bins=np.array(hist_signal.GetXaxis().GetXbins()),
                histtype='step',
                color='blue',
                label="MC Signal"
            )
            # ax.plot(x_fit, y_fit_signal, color='red', linewidth=2, label=f"{sig_pdf_name} fit")
            ax.plot(x_fit, y_fit_signal, color='red', linestyle='--', linewidth=2, label=f"{sig_pdf_name} fit")

        if hist_background:
            hep.histplot(
                np.array([hist_background.GetBinContent(i + 1) for i in range(hist_background.GetNbinsX())]),
                bins=np.array(hist_background.GetXaxis().GetXbins()),
                histtype='step',
                color='green',
                label="MC Background"
            )
            ax.plot(x_fit, y_fit_background, color='purple', linestyle='--', linewidth=2, label=f"bckg {bg_func} fit")
    else:
        hep.histplot(y_values, bins=bins, yerr=y_errors, histtype='errorbar', color='black', label="Data", ax=ax)
        ax.plot(x_fit, y_fit_total, color='red', linestyle='-', linewidth=2, label=f"Fit ({sig_pdf_name})")
    # ax.plot(x_fit, y_fit_signal, color='red', linewidth=2, label=f"{sig_pdf_name} fit")
    # ax.plot(x_fit, y_fit_background, color='purple', linestyle='--', linewidth=2, label="background")


    ax.set_xlabel(var_x_title, fontsize=18)
    ax.set_ylabel("events", fontsize=18)
    ax.legend(fontsize=14)
    # --- Estrazione Parametri e Text Box ---

    mean_val = physical_mean.getVal()
    mean_err = physical_mean.getError()
    if bg_func == "Erf_conv_Exp":
        bg_params_str = '\n'.join((
            f'Bg Model: {bg_pdf_name}',
            r'$m_0 = %.4f \pm %.4f$' % (m0.getVal(), m0.getError()),
            r'$\sigma_\mathrm{erf} = %.4f \pm %.4f$' % (sigma_erf.getVal(), sigma_erf.getError()),
            r'$\alpha_\mathrm{bg} = %.4f \pm %.4f$' % (bg_alpha.getVal(), bg_alpha.getError())
        ))
    else: # Exponential
        bg_params_str = '\n'.join((
            f'Bg Model: {bg_pdf_name}',
            r'$\alpha_\mathrm{bg} = %.4f \pm %.4f$' % (bg_alpha.getVal(), bg_alpha.getError())
        ))

    # Estrazione di sigma (o gamma/width per Voigtian)

    if fit_func == "Voigtian":
        sigma_val = physical_sigma.getVal()
        sigma_err = physical_sigma.getError()
        sigma_label = r'$\sigma$'
        sigma_res_label = r'$\sigma_{res}$'
    if fit_func == "BW_conv_DCS":
        sigma_val = physical_GammaHalf_BW.getVal()
        sigma_err = physical_GammaHalf_BW.getError()
        sigma_label = r'$\Gamma/2$'
        sigma_res_label = r'$\sigma_{res}$'

    else:
        sigma_val = physical_sigma.getVal()
        sigma_err = physical_sigma.getError()
        sigma_label = r'$\sigma$'

    nsig_val = nsig.getVal()
    nbg_val = nbg.getVal()
    textstr = '\n'.join((
        f'sig model: {sig_pdf_name}',
        f'$\mu$ $= %.4f \pm %.4f$' % (physical_mean.getVal(), physical_mean.getError()),
        f'{sigma_label} $= %.4f \pm %.4f$' % (sigma_val, sigma_err),
        # r'$N_\mathrm{sig} = %.0f$' % nsig_val,
        # r'$N_\mathrm{bkg} = %.0f$' % nbg_val
    ))
    if fit_func == "Voigtian":
        textstr = '\n'.join((
        # f'sig model: {sig_pdf_name}',
        f'$\mu$ $= %.4f \pm %.4f$' % (physical_mean.getVal(), physical_mean.getError()),
        f'{sigma_label} $= %.4f \pm %.4f$' % (sigma_val, sigma_err),
        f'{sigma_res_label} $= %.4f \pm %.4f$' % (res_sigma.getVal(), res_sigma.getError()),

        # f'{sigma_res_label} $= %.4f \pm %.4f$' % (sigma_gauss.getVal(), sigma_gauss.getError≈
        # r'$N_\mathrm{sig} = %.0f$' % nsig_val,
        # r'$N_\mathrm{bkg} = %.0f$' % nbg_val
    ))
    if fit_func == "BW_conv_DCS":
        textstr = '\n'.join((
        # f'sig model: {sig_pdf_name}',
        f'$m_Z$ $= %.4f \pm %.4f$' % (physical_mean.getVal(), physical_mean.getError()),
        f'{sigma_label} $= %.4f \pm %.4f$' % (sigma_val, sigma_err),
        f'{sigma_res_label} $= %.4f \pm %.4f$' % (res_sigma.getVal(), res_sigma.getError()),
        # f'$\mu_{{res}} $$= %.4f \pm %.4f$' % (res_mean.getVal(), res_mean.getError()),
        # r'$N_\mathrm{sig} = %.0f$' % nsig_val,
        # r'$N_\mathrm{bkg} = %.0f$' % nbg_val
    ))
    print(f"fitting parameters")
    print(textstr)
    # Place text box in top-right corner
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(
        0.9, 0.75, textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
    )

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(alpha=0.3)
    if wantLogY:
        ax.set_yscale("log")

    hep.cms.label(ax=ax, lumi=period_dict[f"Run3_{year}"], com="13.6", data=not isMC)

    plt.savefig(f"{out_file_name}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_file_name}.png", bbox_inches="tight")
    print(f"Plot salvato in {out_file_name}.pdf e .png")
    plt.close()
