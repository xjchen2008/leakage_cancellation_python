from __future__ import division
import os
import numpy as np
from matplotlib import pyplot as plt
import binascii
import struct
from scipy.signal.waveforms import chirp
from scipy.signal import butter, lfilter, freqz

j = 1j
c = 3e8 # speed of light
samplerate = 56000000
tapping_RX = 0
#rubish = 56000#22400
#nsamples = 56000


def read_tx():
    '''
    num = 4
    for k in range(4):
        num +=2
        signal = np.fromfile('usrp_samples4097_chirp_28MHz'+str(num)+'.dat',np.short,4097)
        plt.plot(signal)
    '''
    signal = np.fromfile('usrp_samples4096_chirp_28MHz_fixed_delayed.dat', np.short, 4097*2)
    #signal = np.fromfile('usrp_samples.dat', np.short, 4097 * 2)
    print(max(signal))
    plt.plot(signal[0::2])
    plt.plot(signal[1::2])
    plt.show()

def readbin( filename, rubish, nsamples ):
    f = open(filename, 'rb')
    f.seek(-(rubish+nsamples)*8,2) # 1879skip the first sample*8 bytes to get rid of junk only 1648*8 is ok
    float = []
    hex2 = []
    count = 0
    try:#read from the beginning to the end
        while 1:
            for i in range(0,4):
                binary8 = f.read(1)
                hex2.append(binascii.hexlify(binary8))
            if binary8 == "":
                break
            hex8=hex2[3]+hex2[2]+hex2[1]+hex2[0]#change order
            float1=struct.unpack('!f', hex8.decode('hex'))[0]
            float.append(float1)
            del hex2[:]
            count += 1
            if count == (2*nsamples):# read certain number of samples
                print("Read", count/2, "Samples.")
                break
    finally:
        f.close()
    real = np.array(float[0:][::2])  # Odd numbers
    imag = np.array(float[1:][::2])  # Even numbers
    mag = np.sqrt(np.power(imag, 2)+np.power(real, 2))
    rx_sig = real+j*imag
    rx_sig_tap = np.append(rx_sig,np.zeros(tapping_RX))
    print("Length of sig_cpx after tapping",len(rx_sig_tap))
    nfft=len(rx_sig)
    freq = np.fft.fftfreq( nfft, d=1./samplerate )
    mag_f_cpx = np.fft.fft(rx_sig_tap)
    logmag_f_cpx = 20*np.log10(np.abs(mag_f_cpx))-100
#    plt.plot(freq,logmag_f_cpx)
    return rx_sig_tap


def last_pulse(filename, nsamples, rubish):
    fsize = int(os.stat(filename).st_size)
    signal = readbin1(filename, nsamples, fsize, rubish)
    signal_ch0, signal_ch1 = get_slice(signal, 256)
    return signal_ch0, signal_ch1


def read_last_pulse(filename, N, pulse_idx_from_back =0):
    nsamples = N
    channels = 2
    # Read latest pulse
    data_ch0, data_ch1 = last_pulse(filename, nsamples * channels, -2*(pulse_idx_from_back+1)*N) #two channels; pulse_idx_from_back+1 because the last pulse could be not a good chirp, so use the second from the back
    rx_error = np.array(data_ch0[0:N])
    return rx_error


def readbin1(filename, nsamples, fsize, rubish):
    bytespersample = 4
    samplesperpulse = nsamples
    total_samples = fsize / bytespersample
    total_pulse = total_samples / samplesperpulse
    file = open(filename, 'rb')
    file.seek((total_pulse - 1) * bytespersample * samplesperpulse)  # find the last pulse
    x = file.read(bytespersample * samplesperpulse)
    file.close()
    fmt = ('%sh' % (len(x) / 2))  # e.g.  '500h' means 500 shorts
    x_sig = np.array(struct.unpack(fmt, x)).astype(float)  # convert to complex float
    rx_sig = x_sig[0::2] + 1j * x_sig[1::2]
    rx_sig = rx_sig / 32767.0
    return rx_sig

def readbin2(filename, dtype, count, rubish):
    x = np.fromfile(filename, dtype, 2*count)  # read the data into numpy
    real = x[::2]/32768.0
    imag = x[1::2]/32768.0
    rx_sig = real+j*imag
    return rx_sig[rubish:]

def get_slice(data, pulselenth):
    data_ch0 = []
    data_ch1 = []
    for i in range(0,len(data), 2*pulselenth):
        data_ch0[i:i+pulselenth] = data[i:i+pulselenth]
        data_ch1[i:i+pulselenth] = data[i+pulselenth:i + 2*pulselenth]
    return data_ch0, data_ch1


def average (filename, nsamples, times, rubish, num_of_channels, chirp_length):
    if num_of_channels == 1:
        fsize = int(os.stat(filename).st_size / 4)
        signal = readbin2(filename, np.short, fsize, rubish)
        signal_sum=np.zeros(nsamples)

        for i in range(0, times):
            signal_temp = signal_sum + signal[i*nsamples: (i+1)*nsamples]
            signal_sum = map(sum, zip(signal_sum, signal_temp))  # summation
        signal_avg = [x / times for x in signal_sum]
        signal_avg = np.asarray(signal_avg)  # list to array
        return signal_avg
    if num_of_channels == 2:
        fsize = int(os.stat(filename).st_size / 4)
        signal = readbin2(filename, np.short, fsize, rubish)

        signal_ch0, signal_ch1 = get_slice(signal, 256)

        signal_sum = np.zeros(2 * nsamples)
        for i in range(0, times):
            signal_temp = signal_sum + signal[i * 2* nsamples: (i + 1) * 2* nsamples] # Averaging for two channles
            signal_sum = map(sum, zip(signal_sum, signal_temp))  # summation
        signal_avg = [x / times for x in signal_sum]
        signal_avg = np.asarray(signal_avg)  # list to array

        return signal_ch0, signal_ch1


def main():
    read_tx()


if __name__ == "__main__":
    main()
