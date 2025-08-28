import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc

date = '20240104'
wavenumber = ''

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def gaus(x,a,x0,sigma):
    return (a/np.sqrt(sigma))*np.exp(-(x-x0)**2/(2*sigma**2))
    # return a*np.exp(-(x-x0)**2/(2*sigma**2))

def cutAndConcatenate(array):
    array = np.asarray(array)
    array = array/array.max()
    idx = (np.abs(array - 0.5)).argmin()
    # idx = cutAndConcatenate(array)
    y = np.concatenate((array[0:idx], (1- array[0:idx])[::-1]), axis=0)
    y = y[::-1]
    x = np.linspace(0, len(y), len(y))
    return x, y

def error(params, x, expt):
    model = gaus(x, *params)
    error = np.sum((model - expt)**2)
    return error

def sigmoidGaus(beamType, stepSize, linearPos):
    x_h, horizontal = np.loadtxt(fr'C:\Users\aruna\Nextcloud\cyanophenol\supporting\Aruna\Beam diameter measurements\{date}\3333\{beamType}_{stepSize}_{linearPos}.txt', unpack=True)
    x_h, horizontal = cutAndConcatenate(horizontal)


    ph = [max(horizontal), np.median(x_h),1,min(horizontal)] # this is an mandatory initial guess
    popt_h, pcov_h = curve_fit(sigmoid, x_h, horizontal,ph, method='dogbox')
    y_h_sigmoid = sigmoid(x_h, *popt_h)

    array_h_dydx = (x_h[:-1] + x_h[1:]) / 2
    dydx_h = np.diff(y_h_sigmoid)/np.diff(x_h)
    dydx_h = dydx_h/dydx_h.max()

    initial = [0.5,len(x_h)/2,1]
    pbounds = [(-5,5), (-400, 400), (0,20)]

    popt_h_gaus = fmin_tnc(error, initial, approx_grad=True, args=(array_h_dydx,dydx_h),bounds=pbounds, maxfun=1000, disp=0)[0]
    y_h_gaus = gaus(array_h_dydx,*popt_h_gaus)
    y_h_gaus = y_h_gaus/y_h_gaus.max()

    if beamType == 'VIS':
        return array_h_dydx, dydx_h, stepSize, y_h_gaus, popt_h_gaus, linearPos
    else:
        return array_h_dydx, dydx_h, y_h_gaus, popt_h_gaus, linearPos

def plottingComp(stepSize, linearPosIR, linearPosVIS, stagePos):

    plt.rc('font', size=7)
    fig = plt.figure(figsize=(4.25, 4.25))
    ax1 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(212)

    array_h_dydx_VIS, dydx_h_VIS, stepSize, y_h_gaus_VIS, popt_h_gaus_VIS, linearPosVIS = sigmoidGaus('VIS', stepSize, linearPosVIS)
    array_h_dydx_IR, dydx_h_IR, y_h_gaus_IR, popt_h_gaus_IR, linearPosIR = sigmoidGaus('IR', stepSize, linearPosIR)

    # ax1.plot(array_h_dydx_VIS*(stepSize/1000), dydx_h_VIS, marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label = 'derivative')
    ax1.plot(array_h_dydx_VIS*(stepSize/1000), y_h_gaus_VIS[::-1], c='g', label = f'Visible, FWHM: {np.round(((popt_h_gaus_VIS[2]*(stepSize/1000)*2.355)),2)} mm')
    ax1.set_xlabel(r'distance / mm')
    # ax1.text(1, 0.8*dydx_h_VIS.max(), f'FWHM: {np.round(((popt_h_gaus_VIS[2]*(stepSize/1000)*2.355)),2)}', style='italic', bbox={
    #         'facecolor': 'blue', 'alpha': 0.3, 'pad': 10})
    ax1.legend()

    # ax1.plot(array_h_dydx_IR*(stepSize/1000), dydx_h_IR, marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label = 'derivative')
    ax1.plot(array_h_dydx_IR*(stepSize/1000), y_h_gaus_IR[::-1], c='r', label = f'IR, FWHM: {np.round(((popt_h_gaus_IR[2]*(stepSize/1000)*2.355)),2)} mm')
    ax1.set_xlabel(r'distance / mm')
    # ax1.text(1, 0.8*dydx_h_IR.max(), f'FWHM: {np.round(((popt_h_gaus_IR[2]*(stepSize/1000)*2.355)),2)}', style='italic', bbox={
    #         'facecolor': 'blue', 'alpha': 0.3, 'pad': 10})
    ax1.legend()

    fig.set_tight_layout(True)
    plt.savefig(fr'C:\Users\aruna\Nextcloud\cyanophenol\supporting\Aruna\Beam diameter measurements\{date}\3333\{stepSize}_{stagePos}_comp.png', dpi=600)

plottingComp(20,'3.5_7','3.5_5.5',3.5)