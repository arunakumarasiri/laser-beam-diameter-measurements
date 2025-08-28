from testpython import *
import win32com.client
import numpy as np

date = '20240516'
wavenumber = ''

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)
def gaus(x,a,x0,sigma):
    return (a/np.sqrt(sigma))*np.exp(-(x-x0)**2/(2*sigma**2))
    # return a*np.exp(-(x-x0)**2/(2*sigma**2))
def setMotorPos(device_id, pos):
   startpos = test_get_position(lib, device_id)
   test_move(lib, device_id, pos)
   test_wait_for_stop(lib, device_id, 100)
   # lib.close_device(byref(cast(device_id, POINTER(c_int))))
   print(f"{device_id} position set to:{pos}")
def stepScan(device_id, start, end, step):
    setMotorPos(device_id, start)
    for pos in range(start,(end+step),step):
        try:
            test_move(lib, device_id, pos)
            test_wait_for_stop(lib, device_id, 100)
            
            time.sleep(2)
            measurement_labView = labViewLogging()
            time.sleep(2)
            print(pos,measurement_labView)

            pos_list.append(pos)   
            measurement_list.append(measurement_labView)
        except:
            print('Detector froze!, moving back to the initial position')
            setMotorPos(device_id, start)
            break
    lib.close_device(byref(cast(device_id, POINTER(c_int))))
def labViewLogging():
    LabVIEW = win32com.client.Dispatch("Labview.Application")
    VI = LabVIEW.getvireference(r'C:\Users\aruna\Nextcloud\cyanophenol\supporting\Aruna\Beam diameter measurements\Example - Single Measurement.vi')
    VI._FlagAsMethod("Call")
    VI.Call()
    measurements_labView = VI.getcontrolvalue('Measurement')
    return measurements_labView
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
def initiate():
    print("\nOpen device " + repr(open_name))
    device_id = lib.open_device(open_name)
    print("Device id: " + repr(device_id))
    test_info(lib, device_id)
    test_status(lib, device_id)
    startpos = test_get_position(lib, device_id)
    print(f'Current position:{startpos}')
    # test_move(lib, device_id, 20000)
    test_wait_for_stop(lib, device_id, 100)
    test_status(lib, device_id)
    test_serial(lib, device_id)
    print("\nClosing")
    lib.close_device(byref(cast(device_id, POINTER(c_int))))
    print("Done")

initiate()
error_occured = 0
pos_list = []
measurement_list = []

##################################################################################

beamType = 'IR_3150'
stepSize = 20
linearPos = '7.5_5'

# Linear position -- IR_VIS

# stepScan(lib.open_device(open_name),-16000,-12000,stepSize)
stepScan(lib.open_device(open_name),-24000,-19000,stepSize)

##################################################################################

array_pos = np.array(pos_list)
array_measurement = np.array(measurement_list)

with open(fr"C:\Users\aruna\Nextcloud\cyanophenol\supporting\Aruna\Beam diameter measurements\{date}\{beamType}_{stepSize}_{linearPos}_nadeem.txt", 'w') as f:
    for index in range(len(pos_list)):
        f.write(str(array_pos[index]) + " " + str(array_measurement [index])+ "\n")
f.close()