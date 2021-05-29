import coe_wavetable_4096 as coe
import numpy as np
import readcsv

#############
# readosc.py
#############
file_tx = 'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg.npy' # this is without PA
#file_tx = 'data/avg/BPF_TL_3999_indoor_40_60MHz_chirp_N100avg_withPA.npy'#'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg_withPA.npy' # this is with PA TL loopback
#file_rx = 'BPF_Antenna_3999_indoor_40_60MHz_chirp_N100avg_withPA_0516_withcanc1.csv' #'output_1_RF_ch2_3999.csv' # run by readosc.py 'antenna_measure.csv' #'data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_Noavg_measure_PA1_canc.csv'
file_rx = 'BPF_Antenna_499999_indoor_40_60MHz_chirp_withPA_antialiasLPF_0529_1.csv'
#file_rx = 'output_1_RF_ch2_499999_chirp.csv'
#########################
# gradient_decent_E312.py
#########################
N = coe.N
simulation_flag = False #False #True
simulation_filename = file_rx #'data\EQ_file_received_chirp_3999_PA_attenna.csv' # 'BPF_TL_3999_indoor_40_60MHz_chirp_N100avg_withPA.csv' #'output_1_RF_ch2_3999_sine.csv' #'output_1_RF_ch2.csv'  # A pre-recorded file for received signal.
EQ_flag = True
EQ_filename = 'EQ_file_received_chirp_499999_PA_TL_20200524.csv' #'EQ_file_received_chirp.cvs' # 1. setch2 to -50dBm sine wave 2. turn on ch2 3. record EQ file.
y_EQ = readcsv.readcsv(filename='data/'+EQ_filename)
y_sim = 1000 *np.load('BPF_Antenna_499999_indoor_40_60MHz_chirp_N100avg_withPA_0516_withcanc.npy') #1000*np.reshape(readcsv.readcsv(filename=file_rx), [N, 1]) # coe.y_cx_combine.real #1000*np.reshape(readcsv.readcsv(filename=file_rx), [N, 1]) #1000*np.reshape(readcsv.readcsv(filename=simulation_filename), [N, 1])  # 1000*np.reshape(np.load(file_tx), [N, 1]) #1000 * np.load(file=file_rx+'.npy') #


nitt = 5 #12 #20
upsamp_rate = 1
K = 100# 26 # Max delay, filter length, how far you want to cancel
Q = 1 # 10 # Order = 2*Q-1
eta = 0.08
delay_step = 4 # 1 # The step size of time delay for template matrix X in function tx_template()
delay_0 = -200 # 20 # 20 for real measurement using Keysight instrument rader.  #-2000#1000 #-200  # -450 #-480  # The initial cancel location. The initial delay in the matrix X in function tx_template()



