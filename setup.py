import coe_wavetable_4096 as coe
import numpy as np
import readcsv

#############
# readosc.py
#############
#  file_tx = 'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg.npy' # this is without PA
file_tx = 'data/avg/BPF_TL_3999_indoor_40_60MHz_chirp_N100avg_withPA.npy'#'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg_withPA.npy' # this is with PA TL loopback
file_rx = 'BPF_Antenna_3999_indoor_40_60MHz_chirp_N100avg_withPA_0516_withcanc.csv' #'output_1_RF_ch2_3999.csv' # run by readosc.py 'antenna_measure.csv' #'data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_Noavg_measure_PA1_canc.csv'

#########################
# gradient_decent_E312.py
#########################
N = coe.N
simulation_flag = True #False #True
simulation_filename = 'BPF_Antenna_3999_indoor_40_60MHz_chirp_N100avg_withPA_0516_nocanc.csv' #'output_1_RF_ch2_3999_sine.csv' #'output_1_RF_ch2.csv'  # A pre-recorded file for received signal.
simulation_filename = simulation_filename #'data\EQ_file_received_chirp_3999_PA_attenna.csv' # 'BPF_TL_3999_indoor_40_60MHz_chirp_N100avg_withPA.csv' #'output_1_RF_ch2_3999_sine.csv' #'output_1_RF_ch2.csv'  # A pre-recorded file for received signal.
EQ_flag = True
EQ_filename = 'EQ_file_received_chirp_3999_PA_TL.csv' #'EQ_file_received_chirp.cvs' # 1. setch2 to -50dBm sine wave 2. turn on ch2 3. record EQ file.
y_EQ = readcsv.readcsv(filename='data/'+EQ_filename)
nitt = 10#5 #20
N = coe.N
D = 100
y_sim = 1000*np.reshape(readcsv.readcsv(filename=simulation_filename), [N, 1]) #1000*np.reshape(readcsv.readcsv(filename=file_rx), [N, 1]) #1000*np.reshape(readcsv.readcsv(filename=simulation_filename), [N, 1])  # 1000*np.reshape(np.load(file_tx), [N, 1])
nitt = 20 #2#5 #20
upsamp_rate = 1
K = 1000 # Max delay, filter length
Q = 1 # 10 # Order = 2*Q-1
eta = 0.1


delay_step = 1  # The step size of time delay for template matrix X in function tx_template()
delay_0 = -40  # -450 #-480  # The initial delay in the matrix X in function tx_template()
#############
# readosc.py
#############
#  file_tx = 'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg.npy' # this is without PA
file_tx =  'data/avg/BPF_TL_3999_indoor_40_60MHz_chirp_N100avg_withPA.npy'#'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg_withPA.npy' # this is with PA TL loopback
file_rx =  'BPF_TL_3999_indoor_40_60MHz_chirp_N100avg_withPA.csv' #'output_1_RF_ch2_3999.csv' # run by readosc.py 'antenna_measure.csv' #'data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_Noavg_measure_PA1_canc.csv'
delay_step = 4 # 4 # The step size of time delay for template matrix X in function tx_template()
delay_0 = -50 #-200  # -450 #-480  # The initial delay in the matrix X in function tx_template()



y_sim = 1000*np.reshape(readcsv.readcsv(filename=file_rx), [N, 1])# 1000*np.reshape(np.load(file_tx), [N, 1]) #1000*np.reshape(readcsv.readcsv(filename=simulation_filename), [N, 1])