import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fmin_tnc
from matplotlib import pyplot as plt
from scipy import special

date = '20240516'
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

beamType = 'IR_3150'
stepSize = 20
linearPos = '7.5_5'

plt.rc('font', size=7)
fig = plt.figure(figsize=(5., 5.))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

x_h, horizontal = np.loadtxt(fr'C:\Users\aruna\Nextcloud\cyanophenol\supporting\Aruna\Beam diameter measurements\{date}\{beamType}_{stepSize}_{linearPos}_nadeem.txt', unpack=True)
# x_h, horizontal = np.loadtxt(f'{beamType}_{stepSize}_{linearPos}.txt', unpack=True)
x_h, horizontal = cutAndConcatenate(horizontal)


ph = [max(horizontal), np.median(x_h),1,min(horizontal)] # this is an mandatory initial guess
popt_h, pcov_h = curve_fit(sigmoid, x_h, horizontal,ph, method='dogbox')
y_h_sigmoid = sigmoid(x_h, *popt_h)

array_h_dydx = (x_h[:-1] + x_h[1:]) / 2
dydx_h = np.diff(y_h_sigmoid)/np.diff(x_h)
dydx_h = dydx_h/dydx_h.max()

initial = [0.5,len(x_h)/2,1]
pbounds = [(-500,500), (-700, 700), (-400,400)]

popt_h_gaus = fmin_tnc(error, initial, approx_grad=True, args=(array_h_dydx,dydx_h),bounds=pbounds, maxfun=1000, disp=0)[0]
y_h_gaus = gaus(array_h_dydx,*popt_h_gaus)
y_h_gaus = y_h_gaus/y_h_gaus.max()

ax1.plot(x_h*(stepSize/1000), horizontal, marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label = 'raw data') 
ax1.plot(x_h*(stepSize/1000), y_h_sigmoid, c='k', label = 'sigmoid fit')
# ax1.plot(x_h*(stepSize/1000), y_h_erf, c='r', label = 'erf fit')
ax1.set_title(f'{beamType}')
ax1.set_ylabel(r'energy / J')
ax1.set_xlabel(r'distance / mm')
ax1.legend()

ax2.plot(array_h_dydx*(stepSize/1000), dydx_h, marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label = 'derivative')
ax2.plot(array_h_dydx*(stepSize/1000), y_h_gaus, c='k', label = 'gaussian fit')
ax2.set_xlabel(r'distance / mm')
ax2.text(0.1, 0.8*dydx_h.max(), f'FWHM: {np.round(((popt_h_gaus[2]*(stepSize/1000)*2.355)),2)} mm', style='italic', bbox={
        'facecolor': 'blue', 'alpha': 0.3, 'pad': 10})
ax2.legend()

fig.set_tight_layout(True)
plt.savefig(fr'C:\Users\aruna\Nextcloud\cyanophenol\supporting\Aruna\Beam diameter measurements\{date}\{beamType}_{stepSize}_{linearPos}_nadeem.png', dpi=600)
# plt.savefig(f'plot.png', dpi=600)

print(popt_h_gaus)
print((popt_h_gaus[2]*(stepSize/1000)*2.355))