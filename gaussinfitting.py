from testpython import *
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from compVISandIR import *
# import pylab

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def gaus(x,a,x0,sigma):
    # return (a/np.sqrt(sigma))*np.exp(-(x-x0)**2/(2*sigma**2))
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

stepSize = 25
linearPos = '0'
beamType = 'IR'


plt.rc('font', size=7)
fig = plt.figure(figsize=(5., 5.))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


x_h, horizontal = np.loadtxt(fr'C:\Users\aruna\Nextcloud\aNewBeginning\supporting\Aruna\Beam diameter measurements\00 Data\{beamType}_{stepSize}_{linearPos}.txt', unpack=True)
horizontal = horizontal[::-1]
# horizontal = horizontal/horizontal.max()

x_h = np.linspace(0,len(x_h), len(x_h))
print(len(x_h))

ph = [max(horizontal), np.median(x_h),1,min(horizontal)] # this is an mandatory initial guess
popt_h, pcov_h = curve_fit(sigmoid, x_h, horizontal,ph, method='dogbox')
y_h_sigmoid = sigmoid(x_h, *popt_h)

array_h_dydx = (x_h[:-1] + x_h[1:]) / 2

dydx_h = np.diff(y_h_sigmoid)/np.diff(x_h)

mean_h = 30
sigma_h = 10

popt_h_gaus,pcov_h_gaus = curve_fit(gaus,array_h_dydx,dydx_h,p0=[1,mean_h,sigma_h])
y_h_gaus = gaus(array_h_dydx, *popt_h_gaus)


ax1.plot(x_h*(stepSize/1000), horizontal, marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label = 'raw data') 
ax1.plot(x_h*(stepSize/1000), y_h_sigmoid, c='k', label = 'sigmoid fit')
ax1.set_title('Horizontal')
ax1.set_ylabel(r'IR energy / J')
ax1.set_xlabel(r'distance / mm')
ax1.legend()

ax2.plot(array_h_dydx*(stepSize/1000), dydx_h, marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label = 'derivative')
ax2.plot(array_h_dydx*(stepSize/1000), y_h_gaus, c='k', label = 'gaussian fit')
ax2.set_xlabel(r'distance / mm')
ax2.text(1, 0.8*dydx_h.max(), f'FWHM: {np.round(((popt_h_gaus[2]*(stepSize/1000)*2.355)),2)}', style='italic', bbox={
        'facecolor': 'blue', 'alpha': 0.3, 'pad': 10})
ax2.legend()

fig.set_tight_layout(True)
plt.savefig(fr'C:\Users\aruna\Nextcloud\aNewBeginning\supporting\Aruna\Beam diameter measurements\00 Data\{beamType}_{stepSize}_{linearPos}.png', dpi=600)

print(popt_h_gaus)
print((popt_h_gaus[2]*(stepSize/1000)*2.355))
# plottingComp()