import coe_wavetable_4096 as coe


#  file_tx = 'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg.npy' # this is without PA
file_tx = 'data/avg/BPF_TL_500000_indoor_40_60MHz_chirp_N100avg_withPA.npy' # this is with PA TL loopback
file_rx = 'EQ_file_received_chirp.cvs' #'data/avg/BPF_antenna_500000_outdoor_40_60MHz_chirp_Noavg_measure_PA1_canc.csv'
nitt = 10
N = coe.N
D = 100
simulation_flag = False #True
simulation_filename = 'output_1_NoEQ_ch2.csv'  # A pre-recorded file for received signal.
EQ_filename = 'EQ_file_received_chirp.cvs'
delay_step = 1  # The step size of time delay for template matrix X in function tx_template()
delay_0 = 0#-450 #-480  # The initial delay in the matrix X in function tx_template()