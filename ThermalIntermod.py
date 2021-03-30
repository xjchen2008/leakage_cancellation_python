import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import find_peaks

################

# Parameters

################

np.random.seed(0)

a0 = 0

# a1 = 7.86

a1 = 3.1623  # 3.1623 for Gp = 10dB # e.g. a1 = 10 when Gp = 20dB; a1 =10^(Gp/20) where Gp is power gain.

a2 = 0  # 0.6

a3 = -52326  # -100# calculated with a1 = 7.861 and IIP3 -10dBm. -52326

Z0 = 50.0  # [Ohms]

f_L = 1e9  # Leakage signal center frequency [Hz]

fs = 100e9  # Sampling frequency [Hz] For accurate result for FFT, fs>300*f_L

N = 1000000

n = 10  # Number of leakage signal power levels. Number of Pin sweep points

timestep = 1.0 / fs

Ts = N * timestep

time = np.linspace(1, Ts, N)

freq = np.fft.fftfreq(N, d=timestep)

P_leak_avg_dBm = np.linspace(-150, -20, n)  # [dBm]

##################################

# Input Thermal Noise

##################################

Pn = (10 ** (-54 / 10)) / 1000  # [Watt] Thermal Noise Power at input = -174dBm = -174dBm/Hz with 1Hz BW

Vn_rms = np.sqrt(Pn * 4 * Z0)  # [Volt]

mean = 0

sigma = Vn_rms

Vn0 = np.random.normal(mean, sigma, N)  # [Volt] Normal distr for white noise as thermal noise

Vn = np.repeat(Vn0, n, axis=0)  # Repeat Vn for n times;

Vn = Vn.reshape(len(Vn0), n)

Vn = np.transpose(Vn)

print(Vn.shape)

# Vn = Vn_peak #[Volt]

V_rms_vrfy = np.sqrt(np.sum(np.power(Vn[0], 2)) / (N - 1))  # [V] RMS definition

Pn_vrfy = 10 * np.log10(V_rms_vrfy ** 2 / (4 * Z0) * 1000)  # [dBm] From Pozar book P499

print(Pn_vrfy)

# Pn_th = np.random.normal(mean,sig_std,N)

Sv_ni = np.fft.fft(Vn) / np.sqrt(
    N)  # #FFT processing gain https://www.edaboard.com/showthread.php?262538-Why-and-how-does-FFT-introduce-a-processing-gain [Volt/Hz] Voltage Spectrum density of the original thermal noise "assuming that the noise samples are not correlated, they add noncoherently and its contribution to the amplitude of Xk increases sqrt(2) times, i.e. power increases 2 times or 3 dB."

Sp_ni = np.power(np.abs(Sv_ni), 2) / (4 * Z0)  # [W/Hz]

Sp_ni_dBmPerHz = 10 * np.log10(Sp_ni[0] * 1000)

Sp_ni_avg = 10 * np.log10(np.sum(Sp_ni[0] * 1000) / N)  # [dBm/Hz] Averaged Sp_ni

print(Sp_ni_avg)

# S_ni = np.fft.fft(Pn_th) # [Volt/Hz] Voltage Spectrum density of the original thermal noise

plt.figure()

plt.plot(freq / 1e9, Sp_ni_dBmPerHz, '.')

plt.xlabel('Frequency [GHz]')

plt.ylabel('Input Thermal Noise Power Density [dBm/Hz]')

plt.title('Input Thermal Noise S_avg(f)=-174dBm/Hz')

##################################

# Input Leakage Signal

##################################


A0 = np.sqrt(10 ** (P_leak_avg_dBm / 10) / 1000 * 2 * Z0)  # [Volt] peak voltage of the leak sig, Power = Vrms * Irms

A = np.repeat(A0, len(time),
              axis=0)  # Repeat A0 for len(time) = N times; for one amplitude, "A" keeps the same value at any time point

A = A.reshape(len(A0), len(time))

t = np.repeat(time, len(A0))

t = t.reshape(len(time), len(A0))

t = np.transpose(t)

# print(A.shape, t.shape)

V_L = A * (np.cos(2 * np.pi * f_L * t))  # This is the leakage signal in time domain

# P_L = np.power(np.abs(V_L),2)/(2*Z0)

Sv_si_source = np.fft.fft(V_L) / N  # [Volt/Hz] Voltage Spectrum Density for leakage signal

Sp_si_source = np.abs(Sv_si_source) ** 2 / (
            2 * Z0)  # [W/Hz] Not same as noise. The averaged power density dilivered to load with rms voltage. Read pozar book P559 and P499

Sp_si_source_dBm = 10 * np.log10(Sp_si_source * 1000)  # [dBm/Hz]

print('The leakage tone power should be 6dB lower then leakage signal Pin'

      'because voltage in right hand of FFT is in half relative to the '

      'amplitude of a cosine wave.\n', 'The Corrected leakage signal power is', max(Sp_si_source_dBm[0]) + 6,
      'dBm in FFT')

plt.figure()

plt.plot(t[0, :] / 1e6, V_L[0, :])

plt.title('Time Domain Leakage %s dBm Signal' % P_leak_avg_dBm[0])

plt.xlabel(' Time [us]')

plt.ylabel('Peak Voltage [V]')

fig = plt.figure()

ax = fig.add_subplot(111)

plt.plot(freq, Sp_si_source_dBm[0, :])

plt.xlabel('Frequency [GHz]')

plt.ylabel('Power Density [dBm/Hz]')

plt.title('Power Spectrum for Input Signal')

A = f_L

B = max(Sp_si_source_dBm[0])

ax.annotate('(%s [GHz], %s [dBm])' % (A / 1e9, int(B)),

            xy=(A, B), xycoords='data',

            xytext=(15, -25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05),

            horizontalalignment='right', verticalalignment='bottom')

Sv_si_port = np.fft.fft(V_L / 2) / N  # Voltage dilivered to LNA input port Read pozar book P559 and P499

Sp_si_port = np.abs(Sv_si_port) ** 2 / (2 * Z0)

'''

fc = 1e9

x = 1 * np.cos(2*np.pi*fc*time)

X = np.fft.fft(x)/N # processing gain of N

X_dBm = 10*np.log10(X**2/Z0*1000)

plt.figure()

plt.plot(freq,X)

'''

##################################

# Output Leakage Signal

##################################

V_L_o = a0 + a1 * (V_L / 2) + a2 * (V_L / 2) ** 2 + a3 * (
            V_L / 2) ** 3  # intermodulation due to nonlinearity; The signal power can be seen as the mean value according to SNR definition https://en.wikipedia.org/wiki/Signal-to-noise_ratio

Vn_o = a0 + a1 * (V_L / 2 + Vn / 2) + a2 * (V_L / 2 + Vn / 2) ** 2 + a3 * (
            V_L / 2 + Vn / 2) ** 3  # intermodulation due to nonlinearity; Thermal noise voltage diveded by 2 because the noise voltage at input port of LNA is half of noise voltage before source impedance. Read pozar book P559 and P499

# Po = np.abs(Vo)**2/(2*Z0) # time domain instantanous power. Useless.

# Sv_no = np.fft.fft(Vn_o)/np.sqrt(N)

Sv_no = np.fft.fft(Vn_o) / np.sqrt(N)  # Noise noncoherent adding. Divided by processing gain

Sp_no = np.abs(Sv_no) ** 2 / (2 * Z0)  # averaged output power spectrum density

Sp_no_dBm = 10 * np.log10(Sp_no * 1000)  # [dBm/Hz]

Sv_so = np.fft.fft(V_L_o) / (N)  # Signal coherent adding

Sp_so = np.abs(Sv_so) ** 2 / (2 * Z0)  # averaged output power spectrum density

# Sp_so = Sp_no

fig = plt.figure()

ax = fig.add_subplot(111)

# plt.plot(freq, Sp_no_dBm[9,:]) #check if signal side lobe dominant noise floor

plt.plot(freq, Sp_no_dBm[9, :])  # check if signal side lobe dominant noise floor

plt.title('Power Spectrum for Output Noise')

indexes1 = find_peaks(Sp_no_dBm[9, :], height=(40, 60), distance=984565)

f_L_idx = indexes1[0]

f_N_idx = int(N * 1 / 2 - 100)  # far from leakge signal frequency

A1 = freq[f_L_idx]  # freq[f_L_idx]

B1 = Sp_no_dBm[9, indexes1[0]]

ax.annotate('(%s [GHz], %s [dBm])' % (A1 / 1e9, int(B1)),

            xy=(A1, B1), xycoords='data',

            xytext=(15, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05),

            horizontalalignment='right', verticalalignment='bottom')

f_N = freq[f_N_idx]  # Test thermal noise frequency

A2 = f_N

B2 = Sp_no_dBm[0, int(f_N_idx)]

ax.annotate('(%s [GHz], %s [dBm])' % (A2 / 1e9, int(B2)),

            xy=(A2, B2), xycoords='data',

            xytext=(15, 25), textcoords='offset points',

            arrowprops=dict(facecolor='black', shrink=0.05),

            horizontalalignment='right', verticalalignment='bottom')

'''

##################################

# Final Result Thermal Noise Power Vs Input Leakage Signal Power

##################################

plt.figure()

plt.plot(P_leak_avg_dBm, Sp_no_dBm[:,f_N_idx]+10*np.log10(N)) # Not averaged, this is not the power. Correction for uncoherent sumation when doing FFT for noise

plt.title('Output Thermal Noise Power Spectrum Point Vs Input Leakage Signal Average Power')

plt.xlabel('Leakage Average Power [dBm]')

plt.ylabel('Thermal Noise Power Spectrum Density [dBm/Hz]')

'''

##################################

# Final Result Thermal Noise Figure Vs Input Leakage Signal Power

##################################

delN = 1000  # numboer of point for noise bandwidth

BW_n = freq[delN]  # noise band width assuming 100 [MHz]

Sp_ni_avg_band = np.sum(Sp_ni[:, f_N_idx - delN:f_N_idx + delN], axis=1) / (2 * delN)  # [dBm/Hz] Averaged Sp_ni

Sp_ni_avg_band_dBm = 10 * np.log10(Sp_ni_avg_band * 1000)

print(Sp_ni_avg_band_dBm)

Sp_no_avg_band = np.sum(Sp_no[:, f_N_idx - delN:f_N_idx + delN], axis=1) / (2 * delN)  # [dBm/Hz] Averaged Sp_no

Sp_no_avg_band_dBm = 10 * np.log10(Sp_no_avg_band * 1000)

print(Sp_no_avg_band_dBm)

P_si = 10 * np.log10(1000 * Sp_si_port[:,
                            f_L_idx].flatten())  # [dBm] Assume signal is a within 1 Hz bandwidth. set the f_L as the freq to calculate power force into 1d arrary

P_ni = 10 * np.log10(1000 * Sp_ni_avg_band * BW_n)  # [dBm] Assume The signal side lobe not dominant the noise floor

P_so = 10 * np.log10(1000 * Sp_so[:, f_L_idx].flatten())  # [dBm]

P_no = 10 * np.log10(1000 * Sp_no_avg_band * BW_n)  # [dBm]

print(P_si.shape, P_ni.shape, P_so.shape, P_no.shape)

print(P_si, P_ni, P_so, P_no)

SNRi_dB = P_si - P_ni  # All power is in dBm

SNRo_dB = P_so - P_no

NF = SNRi_dB - SNRo_dB + 3

plt.figure()

plt.plot(P_leak_avg_dBm, NF)  # Correction for uncoherent sumation when doing FFT for noise

plt.title('Thermal Noise Figure Vs Input Leakage Signal Average Power')

plt.xlabel('Leakage Average Power [dBm]')

plt.ylabel('Noise Figure of LNA [dB]')

##################################

# Final Result Gain Vs Input Leakage Signal Power

##################################

Gain_p = P_so - P_si

plt.figure()

plt.plot(P_leak_avg_dBm, Gain_p)

plt.title('Gain () V.S. Pin')

plt.figure()

plt.plot(P_leak_avg_dBm, P_si)

plt.title('P_si V.S. Pin')

plt.figure()

plt.plot(P_leak_avg_dBm, P_ni)

plt.title('P_ni, Input Thermal Noise Power with Bandwidth = %sMHz' % int(BW_n / 1e6))

plt.xlabel('Leakage Signal Average Power [dBm]')

plt.ylabel('Input Thermal Noise Power [dBm]')

plt.figure()

plt.plot(P_leak_avg_dBm, P_so)

plt.title('P_so  V.S. Pin')

plt.figure()

plt.plot(P_leak_avg_dBm, P_no)

plt.title('P_no, Output Thermal Noise Power with Bandwidth = %sMHz' % int(BW_n / 1e6))

plt.xlabel('Leakage Signal Average Power [dBm]')

plt.ylabel('Output Thermal Noise Power [dBm]')
plt.show()