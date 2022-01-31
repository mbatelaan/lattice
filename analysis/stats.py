# -*- eval: (comment-tags-mode) -*-
import numpy as np
import math
import time as tm
import scipy.optimize as syopt
from scipy.special import gamma, gammainc, gammaincc
import matplotlib.pyplot as pypl
from multiprocessing import Pool

from . import fitfunc as ff

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}
_colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
# _colors = ["r", "g", "b", "k", "y", "m", "k", "k"]
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]


def sortmompar(params):
    amplitudes = params[::2]
    energies = params[1::2]
    sortinglist = [x for _, x in sorted(zip(energies, np.arange(len(energies))))]
    sortedlist = np.array([[amplitudes[i], energies[i]] for i in sortinglist]).flatten()
    return sortedlist


def bs_effmass(data, time_axis=1, spacing=1, a=0.074, plot=False):
    """ Return the effective mass of the data"""
    effmass_ = np.log(np.abs(data[:, :-spacing] / data[:, spacing:])) / spacing
    if plot:
        xlim = 30
        time = np.arange(0, len(data))
        efftime = time[:-1] + 0.5
        yavg = np.average(effmass_, axis=0)
        yerr = np.std(effmass_, axis=0)
        pypl.figure("effmass_plot", figsize=(9, 6))
        # pypl.plot(efftime[:xlim], effmass_[:xlim])
        pypl.xlim(0, xlim)
        pypl.errorbar(
            efftime[:xlim],
            yavg[:xlim],
            yerr[:xlim],
            fmt=".",
            capsize=4,
            elinewidth=1,
            color="k",
            markerfacecolor="none",
        )
        pypl.show()
        pypl.close()
    return effmass_


def effmass(data, a=0.074, spacing=1):
    effmass = np.log(np.abs(data[:-spacing] / data[spacing:])) / spacing
    return effmass


def effamp(data, plot=False, timeslice=10):
    """
    Return the effective amplitude and plot it if plot==True
    Idea comes from Hoerz2020 paper
    """
    effmass0 = bs_effmass(data)
    # effamp = np.abs(
    #     data[:, :-1]
    #     * np.exp(effmass0 * np.arange(len(effmass0[0])), dtype=np.longdouble)
    # )
    effamp = np.abs(
        data[:, :-1]
        * np.exp(effmass0 * np.arange(len(effmass0[0]))))

    if plot:
        xlim = 30
        time = np.arange(0, len(data))
        efftime = time[:-1] + 0.5
        yavg = np.average(effamp, axis=0)
        yerr = np.std(effamp, axis=0)
        pypl.figure("effampplot", figsize=(9, 6))
        # pypl.plot(efftime[:xlim], effamp[:xlim])
        pypl.xlim(0, xlim)
        pypl.errorbar(
            efftime[:xlim],
            yavg[:xlim],
            yerr[:xlim],
            fmt=".",
            capsize=4,
            elinewidth=1,
            color="k",
            markerfacecolor="none",
        )
        pypl.semilogy()
        pypl.show()
        pypl.close()
    return effamp


def fitweights(dof, chisq, derrors):
    """Take a list of degrees of freedom and of chi-squared (not reduced) values and errors of the fit and return the weights for each fit"""
    pf = []
    for d, chi in zip(dof, chisq):
        # pf.append(gammaincc(d/2,chi/2)/gamma(d/2))
        pf.append(gammaincc(d / 2, chi / 2))
    denominator = sum(np.array(pf) * np.array([d ** (-2) for d in derrors]))
    weights = []
    for p, de in zip(pf, derrors):
        weights.append(p * (de ** (-2)) / denominator)
    return weights


def beane_weights(fitlist, param_index=1):
    """Take a list of degrees of freedom and of chi-squared (not reduced) values and errors of the fit and return the weights for each fit"""
    doflist = np.array([i["dof"] for i in fitlist])
    chisqlist = np.array([i["chisq"] for i in fitlist])  # * doflist
    errorlist = np.array([np.std(i["param"], axis=0)[param_index] for i in fitlist])

    pf = gammaincc(doflist / 2, chisqlist / 2)
    denominator = sum(pf * errorlist ** (-2))
    weights = pf * (errorlist ** (-2)) / denominator
    for i, elem in enumerate(fitlist):
        elem["weight"] = weights[i]
    return fitlist, weights


def bayesian_weights(fitlist):
    """
    Calculate the weight of each fit according to the Bayesian formalism explained in the paper by Jay et al (2020)

    fitlist is a list of dictionaries where each dictionary contains the entries chisq, param, x,y and paramavgm"""

    chisq_list = np.array([i["chisq"] for i in fitlist])
    # val_list = np.array([np.average(i["param"], axis=0)[1] for i in fitlist])
    # error_list = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
    Ncut = np.array([np.shape(fitlist[-1]["y"])[1] - len(i["x"]) for i in fitlist])
    Ncut = Ncut - np.min(Ncut)
    parnum = np.array([len(i["paramavg"]) for i in fitlist])
    AIC1 = (
        -2 * np.log(1 / (4 * len(fitlist)))
        + np.array(chisq_list)
        + 2 * parnum
        + 2 * Ncut
    )
    prMD = np.exp(-2 * AIC1)
    # Normalize probabilities
    prMDsum = np.sum(prMD)  # +np.sum(prMD2)+np.sum(prMD3)
    prMD = prMD / prMDsum
    return prMD


def fitratio(fitfnc, p0, x, data, bounds=None, time=False, fullcov=False):
    """
    Fit to every bootstrap ensemble and use multiprocessing to split up the task over two processors
    p0: initial guess for the parameters
    x: array of x values to fit over
    data: array/list of BootStrap objects
    """
    if time:
        start = tm.time()

    nboot = np.shape(data)[0]
    yerr = np.std(data, axis=0)
    # cvinv = np.diag(np.ones(len(data[0])))
    # cvinv = np.linalg.inv(np.diag(yerr ** 2))
    cvinv = np.linalg.inv(np.cov(data.T))
    dataavg = np.average(data, axis=0)

    ### Fit to the data avg with the full covariance matrix
    cvinv = np.linalg.inv(np.cov(data.T))
    resavg = syopt.minimize(
        ff.chisqfn,
        p0,
        args=(fitfnc, x, dataavg, cvinv),
        method="Nelder-Mead",
        # bounds=bounds,
        options={"disp": False},
    )
    redchisq = resavg.fun / (len(data[0, :]) - len(p0))
    p0 = resavg.x
    # print(f"{resavg.x}")
    # print(f"{redchisq}")

    param_bs = np.zeros((nboot, len(p0)))
    cvinv = np.linalg.inv(np.diag(yerr ** 2))
    for iboot in range(nboot):
        yboot = data[iboot, :]
        res = syopt.minimize(
            ff.chisqfn,
            p0,
            args=(fitfnc, x, yboot, cvinv),
            method="Nelder-Mead",
            # method="L-BFGS-B",
            # bounds=bounds,
            options={"disp": False},
        )
        param_bs[iboot] = res.x

    fitparam = {
        "x": x,
        "y": data,
        "fitfunction": fitfnc,
        "paramavg": resavg.x,
        "param": param_bs,
        "redchisq": redchisq,
        "dof": len(x) - len(p0),
    }
    if time:
        print("fitratio time: \t", tm.time() - start)
    return fitparam


def fit_bootstrap(fitfnc, p0, x, data, bounds=None, time=False, fullcov=False):
    """
    Fit to every bootstrap ensemble and use multiprocessing to split up the task over two processors
    p0: initial guess for the parameters
    x: array of x values to fit over
    data: array/list of BootStrap objects
    """
    if time:
        start = tm.time()

    nboot = np.shape(data)[0]
    yerr = np.std(data, axis=0)
    # cvinv = np.diag(np.ones(len(data[0])))
    # cvinv = np.linalg.inv(np.diag(yerr ** 2))
    cvinv = np.linalg.inv(np.cov(data.T))
    dataavg = np.average(data, axis=0)

    ### Fit to the data avg with the full covariance matrix
    resavg = syopt.minimize(
        ff.chisqfn,
        p0,
        args=(fitfnc, x, dataavg, cvinv),
        method="Nelder-Mead",
        # bounds=bounds,
        options={"disp": False},
    )
    chisq = resavg.fun
    redchisq = resavg.fun / (len(data[0, :]) - len(p0))
    p0 = resavg.x
    # print(f"{resavg.x}")
    # print(f"{redchisq}")

    param_bs = np.zeros((nboot, len(p0)))
    # cvinv = np.linalg.inv(np.diag(yerr ** 2))
    for iboot in range(nboot):
        yboot = data[iboot, :]
        res = syopt.minimize(
            ff.chisqfn,
            p0,
            args=(fitfnc, x, yboot, cvinv),
            method="Nelder-Mead",
            # method="L-BFGS-B",
            # bounds=bounds,
            options={"disp": False},
        )
        param_bs[iboot] = res.x

    fitparam = {
        "x": x,
        "y": data,
        "fitfunction": fitfnc,
        "paramavg": resavg.x,
        "param": param_bs,
        "chisq": chisq,
        "redchisq": redchisq,
        "dof": len(x) - len(p0),
    }
    if time:
        print("fit_bootstrap time: \t", tm.time() - start)
    return fitparam


def fit_bootstrap_bayes(
    fitfnc, p0, prior, priorsigma, x, data, bounds=None, time=False
):
    """
    Fit to every bootstrap ensemble using the bayesian fitting methods
    p0: initial guess for the parameters
    x: array of x values to fit over
    data: array/list of BootStrap objects
    """
    if time:
        start = tm.time()

    nboot = np.shape(data)[0]
    yerr = np.std(data, axis=0)
    # Idinv = np.diag(np.ones(len(data[0])))
    varinv = np.linalg.inv(np.diag(yerr ** 2))
    cvinv = np.linalg.inv(np.cov(data.T))
    dataavg = np.average(data, axis=0)

    ### Fit to the data avg with the full covariance matrix
    resavg = syopt.minimize(
        ff.chisqfn_bayes,
        p0,
        args=(fitfnc, x, dataavg, varinv, prior, priorsigma),
        method="Nelder-Mead",
        # bounds=bounds,
        options={"disp": False},
    )
    chisq = ff.chisqfn_bayes(resavg.x, fitfnc, x, dataavg, cvinv, prior, priorsigma)
    # redchisq = resavg.fun / (len(dataavg) - len(p0))
    redchisq = chisq / (len(dataavg) - len(p0))
    # p0 = resavg.x
    # print(f"{resavg.x}")
    # print(f"{redchisq}")

    param_bs = np.zeros((nboot, len(p0)))
    # cvinv = np.linalg.inv(np.diag(yerr ** 2))
    for iboot in range(nboot):
        yboot = data[iboot, :]
        res = syopt.minimize(
            ff.chisqfn_bayes,
            p0,
            args=(fitfnc, x, yboot, varinv, prior, priorsigma),
            method="Nelder-Mead",
            # method="L-BFGS-B",
            # bounds=bounds,
            options={"disp": False},
        )
        param_bs[iboot] = res.x

    fitparam = {
        "x": x,
        "y": data,
        "fitfunction": fitfnc,
        "paramavg": resavg.x,
        "param": param_bs,
        "redchisq": redchisq,
        "chisq": chisq,
        "dof": len(x) - len(p0),
    }
    if time:
        end = tm.time()
        print("fit_bootstrap_bayes time: \t", end - start)
        print("fits per second: \t", (nboot + 1) / (end - start))
    return fitparam


def fit_bootstrap_ratio(
    fitfnc,
    fitfnc_ratio,
    x,
    x_energy,
    data,
    data_ratio1,
    data_ratio2,
    p0,
    p0_q1,
    p0_q2,
    time=False,
    disp=False,
    fullcov=False,
):
    """
    Fit to every bootstrap ensemble and use multiprocessing to split up the task over two processors
    p0: initial guess for the parameters
    x: array of x values to fit over
    data: array/list of BootStrap objects
    """
    if time:
        start = tm.time()

    nboot = np.shape(data)[0]
    yerr = np.std(data, axis=0)
    # cvinv = np.linalg.inv(np.cov(data.T))
    cvinv = np.linalg.inv(np.diag(np.std(data, axis=0) ** 2))
    dataavg = np.average(data, axis=0)

    ### Fit to the data avg two-point function
    resavg = syopt.minimize(
        ff.chisqfn,
        p0,
        args=(fitfnc.eval, x_energy, dataavg, cvinv),
        method="Nelder-Mead",
        # bounds=bounds,
        options={"disp": False},
    )
    chisq = resavg.fun
    redchisq = resavg.fun / (len(data[0, :]) - len(p0))
    p0 = resavg.x

    # Set the q parameters for the ratio fit
    fitfnc_ratio.q = resavg.x

    ### Quark 1 ratio fit
    # cvinv_ratio1 = np.linalg.inv(np.cov(data_ratio1.T))
    # cvinv_ratio1 = np.linalg.inv(np.cov(data_ratio1.T))
    cvinv_ratio1 = np.linalg.inv(np.diag(np.std(data_ratio1, axis=0) ** 2))
    dataavg_ratio1 = np.average(data_ratio1, axis=0)
    resavg_ratio1 = syopt.minimize(
        ff.chisqfn,
        p0_q1,
        args=(fitfnc_ratio.eval, x, dataavg_ratio1, cvinv_ratio1),
        method="Nelder-Mead",
        # bounds=bounds,
        options={"disp": False},
    )
    chisq_ratio1 = resavg_ratio1.fun
    redchisq_ratio1 = resavg_ratio1.fun / (
        len(data_ratio1[0, :]) - len(fitfnc_ratio.initpar)
    )
    p0_q1 = resavg_ratio1.x

    ### Quark 2 ratio fit
    # cvinv_ratio2 = np.linalg.inv(np.cov(data_ratio2.T))
    cvinv_ratio2 = np.linalg.inv(np.diag(np.std(data_ratio2, axis=0) ** 2))
    dataavg_ratio2 = np.average(data_ratio2, axis=0)
    resavg_ratio2 = syopt.minimize(
        ff.chisqfn,
        p0_q2,
        args=(fitfnc_ratio.eval, x, dataavg_ratio2, cvinv_ratio2),
        method="Nelder-Mead",
        # bounds=bounds,
        options={"disp": False},
    )
    chisq_ratio2 = resavg_ratio2.fun
    redchisq_ratio2 = resavg_ratio2.fun / (
        len(data_ratio2[0, :]) - len(fitfnc_ratio.initpar)
    )
    p0_q2 = resavg_ratio2.x
    # print(f"{resavg_ratio2.x}")

    ### Fit to each bootstrap resample
    param_bs = np.zeros((nboot, len(p0)))
    param_bs_q1 = np.zeros((nboot, len(fitfnc_ratio.initpar)))
    param_bs_q2 = np.zeros((nboot, len(fitfnc_ratio.initpar)))
    for iboot in range(nboot):
        # 2pt. function data
        yboot = data[iboot, :]
        res = syopt.minimize(
            ff.chisqfn,
            p0,
            args=(fitfnc.eval, x_energy, yboot, cvinv),
            method="Nelder-Mead",
            options={"disp": False},
        )
        if res.x[1] < res.x[3]:
            param_bs[iboot] = res.x
            fitfnc_ratio.q = res.x
        else:
            param_bs[iboot] = np.array([res.x[2], res.x[3], res.x[0], res.x[1]])
            fitfnc_ratio.q = np.array([res.x[2], res.x[3], res.x[0], res.x[1]])

        # Quark 1 ratio
        yboot1 = data_ratio1[iboot, :]
        res1 = syopt.minimize(
            ff.chisqfn,
            p0_q1,
            args=(fitfnc_ratio.eval, x, yboot1, cvinv_ratio1),
            method="Nelder-Mead",
            options={"disp": False},
        )
        param_bs_q1[iboot] = res1.x

        # Quark 2 ratio
        yboot2 = data_ratio2[iboot, :]
        res2 = syopt.minimize(
            ff.chisqfn,
            p0_q2,
            args=(fitfnc_ratio.eval, x, yboot2, cvinv_ratio2),
            method="Nelder-Mead",
            options={"disp": False},
        )
        param_bs_q2[iboot] = res2.x

    fitparam = {
        "x": x_energy,
        "y": data,
        "fitfunction": fitfnc.eval,
        "paramavg": resavg.x,
        "param": param_bs,
        "chisq": chisq,
        "redchisq": redchisq,
        "dof": len(x) - len(p0),
    }
    fitparam_q1 = {
        "x": x,
        "y": data_ratio1,
        "fitfunction": fitfnc_ratio.eval,
        "paramavg": resavg_ratio1.x,
        "param": param_bs_q1,
        "chisq": chisq_ratio1,
        "redchisq": redchisq_ratio1,
        "dof": len(x) - len(p0_q1),
    }
    fitparam_q2 = {
        "x": x,
        "y": data_ratio2,
        "fitfunction": fitfnc_ratio.eval,
        "paramavg": resavg_ratio2.x,
        "param": param_bs_q2,
        "chisq": chisq_ratio2,
        "redchisq": redchisq_ratio2,
        "dof": len(x) - len(p0_q2),
    }
    if time:
        print("fit_bootstrap time: \t", tm.time() - start)
    return fitparam, fitparam_q1, fitparam_q2


def weights(dof, chisq, derrors):
    """
    Take a list of degrees of freedom and of chi-squared values and errors of the fit and return the weights for each fit. This uses the weighting discussed in appendix B of Beane2020.
    """
    pf = gammaincc(dof / 2, chisq / 2)
    denominator = sum(pf * derrors ** (-2))
    weights = pf * (derrors ** (-2)) / denominator
    return weights


def fit_loop(
    data, fitfnc, time_limits, plot=False, disp=False, time=False, weights_=False
):
    """
    Fit the correlator by looping over time ranges and calculating the weight for each fit.

    time_limits = [[tminmin,tminmax],[tmaxmin, tmaxmax]]
    """

    ### Get the effective mass and amplitude for p0
    amp0 = effamp(data, plot=False)
    mass0 = bs_effmass(data, plot=False)

    ### Set the initial guesses for the parameters
    # timeslice = 13
    fitfnc.initparfnc(data)
    # fitfnc.initparfnc(
    #     [np.average(amp0, axis=0)[timeslice], np.std(amp0, axis=0)[timeslice]],
    #     [np.average(mass0, axis=0)[timeslice], np.std(mass0, axis=0)[timeslice]],
    # )

    fitlist = []
    [[tminmin, tminmax], [tmaxmin, tmaxmax]] = time_limits
    for tmin in range(tminmin, tminmax + 1):
        for tmax in range(tmaxmin, tmaxmax + 1):
            if tmax - tmin > len(fitfnc.initpar) + 1:
                timerange = np.arange(tmin, tmax)
                if disp:
                    print(f"\ntime range = {tmin}-{tmax}")
                ydata = data[:, timerange]

                ### Perform the fit
                fitparam_unpert = fit_bootstrap(
                    fitfnc.eval,
                    fitfnc.initpar,
                    timerange,
                    ydata,
                    bounds=fitfnc.bounds,
                    time=time,
                )
                if disp:
                    print(f"parameter values = {fitparam_unpert['paramavg']}")
                    print(f"chi-sq. per dof. = {fitparam_unpert['redchisq']}")
                    # print(f"{fitparam_unpert['paramavg']}")
                    # print(f"{fitparam_unpert['redchisq']}")
                    # print(f"{np.average(fitparam_unpert['param'],axis=0)}\n")
                fitparam_unpert["y"] = data
                fitlist.append(fitparam_unpert)

    if weights_:
        fitlist, weightlist = beane_weights(fitlist, param_index=1)
        # ### Calculate the weights of each of the fits
        # doflist = np.array([i["dof"] for i in fitlist])
        # chisqlist = np.array([i["redchisq"] for i in fitlist]) * doflist
        # errorlist = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
        # weightlist = weights(doflist, chisqlist, errorlist)
        # for i, elem in enumerate(fitlist):
        #     elem["weight"] = weightlist[i]
    return fitlist


def fit_loop_ratio(
    data,
    data_ratio1,
    data_ratio2,
    fitfnc,
    fitfnc_ratio,
    time_limits,
    plot=False,
    disp=False,
    time=False,
    weights_=False,
):
    """
    Fit the correlator by looping over time ranges and calculating the weight for each fit.

    time_limits = [[tminmin,tminmax],[tmaxmin, tmaxmax]]
    """

    ### Get the effective mass and amplitude for p0
    amp0 = effamp(data, plot=False)
    mass0 = bs_effmass(data, plot=False)

    ### Set the initial guesses for the parameters
    fitfnc.initparfnc(data)
    fitfnc_ratio.initparfnc(data_ratio1)
    p0_q1 = fitfnc_ratio.initpar
    fitfnc_ratio.initparfnc(data_ratio2)
    p0_q2 = fitfnc_ratio.initpar

    fitlist = []
    fitlist_q1 = []
    fitlist_q2 = []
    [[tminmin, tminmax], [tmaxmin, tmaxmax]] = time_limits
    for tmin in range(tminmin, tminmax):
        for tmax in range(tmaxmin, tmaxmax):
            if tmax - tmin > len(fitfnc.initpar) + 1:
                timerange = np.arange(tmin, tmax)
                timerange_energy = np.arange(tmin, tmax + 10)
                if disp:
                    print(f"\ntime range = {tmin}-{tmax}")
                ydata = data[:, timerange_energy]
                ydata_q1 = data_ratio1[:, timerange]
                ydata_q2 = data_ratio2[:, timerange]

                fitparam_unpert, fitparam_q1, fitparam_q2 = fit_bootstrap_ratio(
                    fitfnc,
                    fitfnc_ratio,
                    timerange,
                    timerange_energy,
                    ydata,
                    ydata_q1,
                    ydata_q2,
                    fitfnc.initpar,
                    p0_q1,
                    p0_q2,
                    time=time,
                    fullcov=False,
                    disp=disp,
                )
                if disp:
                    print(f"parameter values = {fitparam_unpert['paramavg']}")
                    print(f"chi-sq. per dof. = {fitparam_unpert['redchisq']}")
                    print(f"parameter values = {fitparam_q1['paramavg']}")
                    print(f"chi-sq. per dof. = {fitparam_q1['redchisq']}")
                    print(f"parameter values = {fitparam_q2['paramavg']}")
                    print(f"chi-sq. per dof. = {fitparam_q2['redchisq']}")
                fitparam_unpert["y"] = data
                fitparam_q1["y"] = data_ratio1
                fitparam_q2["y"] = data_ratio2
                fitlist.append(fitparam_unpert)
                fitlist_q1.append(fitparam_q1)
                fitlist_q2.append(fitparam_q2)

    if weights_:
        fitlist, weightlist = beane_weights(fitlist, param_index=1)
        fitlist_q1, weightlist1 = beane_weights(fitlist_q1, param_index=1)
        fitlist_q2, weightlist2 = beane_weights(fitlist_q2, param_index=1)

    return fitlist, fitlist_q1, fitlist_q2


def fit_loop_bayes(
    data, fitfnc, time_limits, plot=False, disp=False, time=False, weights_=False
):
    """
    Fit the correlator by looping over time ranges and calculating the weight for each fit.

    time_limits = [[tminmin,tminmax],[tmaxmin, tmaxmax]]
    """

    ### Get the effective mass and amplitude for p0
    amp0 = effamp(data, plot=False)
    mass0 = bs_effmass(data, plot=False)

    ### Set the initial guesses for the parameters
    # timeslice = 13
    timeslice = time_limits[1][1] - 3
    fitfnc.initparfnc(data, timeslice=timeslice)
    # fitfnc.initparfnc(
    #     [np.average(amp0, axis=0)[timeslice], np.std(amp0, axis=0)[timeslice]],
    #     [np.average(mass0, axis=0)[timeslice], np.std(mass0, axis=0)[timeslice]],
    # )

    fitlist = []
    [[tminmin, tminmax], [tmaxmin, tmaxmax]] = time_limits
    for tmin in range(tminmin, tminmax + 1):
        for tmax in range(tmaxmin, tmaxmax + 1):
            if tmax - tmin > len(fitfnc.initpar) + 1:
                timerange = np.arange(tmin, tmax)
                if disp:
                    print(f"\ntime range = {tmin}-{tmax}")
                ydata = data[:, timerange]

                ### Perform the fit
                fitparam_unpert = fit_bootstrap_bayes(
                    fitfnc.eval,
                    fitfnc.initpar,
                    fitfnc.initpar,
                    fitfnc.priorsigma,
                    timerange,
                    ydata,
                    bounds=None,
                    time=time,
                )
                if disp:
                    print(f"parameter values = {fitparam_unpert['paramavg']}")
                    print(f"chi-sq. per dof. = {fitparam_unpert['redchisq']}")
                    # print(f"{fitparam_unpert['paramavg']}")
                    # print(f"{fitparam_unpert['redchisq']}")
                    # print(f"{np.average(fitparam_unpert['param'],axis=0)}\n")
                fitparam_unpert["y"] = data
                fitlist.append(fitparam_unpert)

    if weights_:
        ### Calculate the weights of each of the fits
        doflist = np.array([i["dof"] for i in fitlist])
        chisqlist = np.array([i["redchisq"] for i in fitlist]) * doflist
        errorlist = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
        weightlist = bayesian_weights(fitlist)
        # weightlist = weights(doflist, chisqlist, errorlist)
        for i, elem in enumerate(fitlist):
            elem["weight"] = weightlist[i]

    return fitlist


def ploteffmass(
    correlator,
    plotname,
    plotdir,
    lmb=1,
    xlim=48,
    ylim=None,
    fitparam=None,
    ylabel=None,
    show=False,
):
    """Plot the effective mass of the correlator and maybe plot their fits"""
    spacing = 2
    time = np.arange(0, np.shape(correlator)[1])
    efftime = time[:-spacing] + 0.5

    effmassdata = bs_effmass(correlator, time_axis=1, spacing=spacing) / lmb
    # effmassdata = correlator[:, :-1]
    yeffavg = np.average(effmassdata, axis=0)
    yeffstd = np.std(effmassdata, axis=0)

    pypl.figure(figsize=(9, 6))
    pypl.errorbar(
        efftime[:xlim],
        yeffavg[:xlim],
        yeffstd[:xlim],
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    if fitparam:
        pypl.plot(
            fitparam["x"],
            np.average(fitparam["y"], axis=0) / lmb,
            label=fitparam["label"],
        )
        pypl.fill_between(
            fitparam["x"],
            (np.average(fitparam["y"], axis=0) - np.std(fitparam["y"], axis=0)) / lmb,
            (np.average(fitparam["y"], axis=0) + np.std(fitparam["y"], axis=0)) / lmb,
            alpha=0.3,
        )
        pypl.legend()

    # if fitparam:
    #     pypl.plot(fitparam[0], np.average(fitparam[1], axis=0))
    #     pypl.fill_between(
    #         fitparam[0],
    #         np.average(fitparam[1], axis=0) - np.std(fitparam[1], axis=0),
    #         np.average(fitparam[1], axis=0) + np.std(fitparam[1], axis=0),
    #         alpha=0.3,
    #     )

    pypl.xlabel(r"$\textrm{t/a}$", labelpad=14, fontsize=18)
    pypl.ylabel(ylabel, labelpad=5, fontsize=18)
    # pypl.ylabel(r'$\Delta E/\lambda$',labelpad=5,fontsize=18)
    # pypl.title(r'Energy shift '+pars.momfold[pars.momentum][:-1]+r', $\gamma_{'+op[1:]+r'}$')

    # if ylim:
    #     print(ylim)
    #     ylim2 = [-1 * i for i in ylim[::-1]]
    #     print(ylim2)
    #     pypl.ylim(ylim2)
    pypl.ylim(ylim)
    pypl.xlim(0, xlim - 1)
    # pypl.grid(True, alpha=0.4)
    # _metadata["Title"] = plotname.split("/")[-1][:-4]
    _metadata["Title"] = plotname
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()
    return


def plot_correlator(
    correlator,
    plotname,
    plotdir,
    lmb=1,
    xlim=48,
    ylim=None,
    fitparam=None,
    ylabel=None,
    show=False,
):
    """Plot the effective mass of the correlator and maybe plot their fits"""
    time = np.arange(0, np.shape(correlator)[1])
    yavg = np.average(correlator, axis=0)
    ystd = np.std(correlator, axis=0)

    pypl.figure(figsize=(9, 6))
    pypl.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    if fitparam:
        pypl.plot(
            fitparam["x"],
            np.average(fitparam["y"], axis=0) / lmb,
            label=fitparam["label"],
        )
        pypl.fill_between(
            fitparam["x"],
            (np.average(fitparam["y"], axis=0) - np.std(fitparam["y"], axis=0)) / lmb,
            (np.average(fitparam["y"], axis=0) + np.std(fitparam["y"], axis=0)) / lmb,
            alpha=0.3,
        )
        pypl.legend()
    pypl.xlabel(r"$\textrm{t/a}$", labelpad=14, fontsize=18)
    pypl.ylabel(ylabel, labelpad=5, fontsize=18)
    pypl.ylim(ylim)
    pypl.xlim(0, xlim - 1)
    _metadata["Title"] = plotname
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()
    return


def plot_fit_effmass(
    fit,
    title,
    save_file=None,
    FH_ratio=None,
    w_avg=None,
    w_avg_val=None,
    xlim=48,
    ylim=None,
    ylabel=None,
    show=False,
):
    """Plot the effective mass of the correlator and maybe plot their fits"""

    ydata = bs_effmass(fit["y"], time_axis=1, spacing=1)
    fitdata = bs_effmass(
        np.array(
            [
                fit["fitfunction"](fit["x"], fit["param"][boot, :])
                for boot in range(len(fit["param"]))
            ]
        ),
        time_axis=1,
        spacing=1,
    )

    if FH_ratio != None:
        fitfnc_ratio = ff.initffncs("TwoexpRatio4")
        fitdata = bs_effmass(
            np.array(
                [
                    fitfnc_ratio.eval2(
                        fit["x"], FH_ratio["param"][boot, :], fit["param"][boot, :]
                    )
                    for boot in range(len(fit["param"]))
                ]
            ),
            time_axis=1,
            spacing=1,
        )

    f, axs = pypl.subplots(1, 1, figsize=(12, 6), sharex=True)
    axs.errorbar(
        np.arange(len(ydata[0])),
        np.average(ydata, axis=0),
        np.std(ydata, axis=0),
        capsize=4,
        elinewidth=1,
        color="k",
        fmt="s",
        markerfacecolor="none",
    )
    axs.fill_between(
        fit["x"][:-1],
        np.average(fitdata, axis=0) + np.std(fitdata, axis=0),
        np.average(fitdata, axis=0) - np.std(fitdata, axis=0),
        alpha=0.5,
        color=_colors[0],
        linewidth=0,
    )
    if w_avg != None:
        axs.fill_between(
            np.arange(len(ydata[0])),
            [np.average(w_avg_val)] * len(ydata[0]),
            [np.average(w_avg_val) + np.std(w_avg_val)] * len(ydata[0]),
            [np.average(w_avg_val) - np.std(w_avg_val)] * len(ydata[0]),
            alpha=0.5,
            color=_colors[1],
            linewidth=0,
        )

    axs.plot(
        fit["x"][:-1],
        np.average(fitdata, axis=0),
        label=r"$\chi^2_{\textrm{dof}} = $" + f"{fit['redchisq']:0.2f}",
        color=_colors[0],
    )

    axs.grid(True, alpha=0.4)
    axs.set_title(title)
    axs.set_xlim(0, 48)
    axs.set_ylim(ylim)
    axs.set_ylabel(ylabel, labelpad=5, fontsize=18)
    axs.set_xlabel(r"$\textrm{t/a}$", labelpad=14, fontsize=18)
    axs.legend()

    _metadata["Title"] = title
    if save_file:
        f.savefig(save_file, metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()
    return


def filter_fitlist_comb(data1, data2, window=1):
    """Filter out one tmax"""
    weightlist1 = np.array([i["weight"] for i in data1])
    weightorder1 = weightlist1.argsort()
    weightlist2 = np.array([i["weight"] for i in data2])
    weightorder2 = weightlist2.argsort()
    tmax1 = data1[weightorder1[-1]]["x"][-1]
    tmax2 = data2[weightorder2[-1]]["x"][-1]
    fitlist1 = []
    for fitdata in data1:
        if min(tmax1, tmax2) - window < fitdata["x"][-1] < max(tmax1, tmax2) + window:
            fitlist1.append(fitdata)
    fitlist2 = []
    for fitdata in data2:
        if min(tmax1, tmax2) - window < fitdata["x"][-1] < max(tmax1, tmax2) + window:
            fitlist2.append(fitdata)

    fitlist1, weightlist1 = stats.beane_weights(fitlist1)
    # for i, elem in enumerate(fitlist1):
    #     elem["weight"] = weightlist1[i]
    fitlist2, weightlist2 = stats.beane_weights(fitlist2)
    # for i, elem in enumerate(fitlist2):
    #     elem["weight"] = weightlist2[i]

    print(np.array([i["weight"] for i in fitlist1]))
    print(np.array([i["weight"] for i in fitlist2]))
    print(np.array([i["chisq"] for i in fitlist1]))
    print(np.array([i["chisq"] for i in fitlist2]))
    return fitlist1, fitlist2


def filter_fitlist(data1, tmax=None, window=1):
    """Filter out one tmax"""
    weightlist1 = np.array([i["weight"] for i in data1])
    weightorder1 = weightlist1.argsort()
    if tmax:
        pass
    else:
        tmax = data1[weightorder1[-1]]["x"][-1]

    fitlist1 = []
    for fitdata in data1:
        if tmax - window < fitdata["x"][-1] < tmax + window:
            fitlist1.append(fitdata)
    return fitlist1
