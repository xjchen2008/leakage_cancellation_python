#!/usr/bin/env python

import visa
from time import sleep
import argparse
import struct
import numpy as np
import csv
import os
import coe_wavetable_4096 as coe
from functions import equalizer
import readosc
import matplotlib.pyplot as plt
import setup


def UploadArb(x_canc=np.zeros(5000), SOUR = 'SOUR1'):
    #parse arguments
    parser = argparse.ArgumentParser(description='Upload an arbitrary waveform to an Agilent 33600 AWG')
    parser.add_argument('-f','--filename', help='File containing arbitrary waveform', default="./ATLASCALIB.dat", required=False)
    parser.add_argument('-a','--address', help="Address of device", default="169.254.5.21", required=False)
    parser.add_argument('-v','--pulseheight', help="Pulse amplitude of arb in dBm", default="10", required=False) #12 6.5 -20.5dBm ch1  = 7dBm is OK with distortion. ch2 -40dBm
    parser.add_argument('-m','--macro', help="Generate a macro for loading this arb", action='store_true', required=False)
    parser.add_argument('-d','--delimiter', help="Input file delimiter", default=" ", required=False)


    args = parser.parse_args()

    #can't pass '\t' in at the command line
    if args.delimiter=="tab":
        args.delimiter="\t"


    #remove file extension
    name=os.path.splitext(os.path.basename(args.filename))[0]

    #if the arb name is longer than 12, truncate it (maximum length allowed by SCPI interface)
    if len(name) > 12:
        name=name[:12]
        print("Arb name truncated to "+name)


    samplePeriod=0
    num=0
    tlast=-1

    #get sample rate
    sRate = str(250000000) #str(250000000)#str(100e6) #250000000

    #scale signal between 1 and -1
    arb = x_canc
    sig = np.asarray(arb, dtype='f4')


    #load the VISA resource manager
    rm = visa.ResourceManager('@py')

    #connect to the device
    inst = rm.open_resource("TCPIP::"+args.address+"::INSTR")
    print(inst.query("*IDN?"))

    #sent start control message
    message="Uploading\nArbitrary\nWaveform"
    inst.write("DISP:TEXT '"+message+"'")

    #create a directory on device (will generate error if the directory exists, but we can ignore that)
    inst.write("MMEMORY:MDIR \"INT:\\remoteAdded\"")

    #set byte order
    inst.write('FORM:BORD SWAP')

    #clear volatile memory
    #inst.write('SOUR2'+':DATA:VOL:CLE')
    inst.write(SOUR + ':DATA:VOL:CLE')

    #write arb to device
    inst.write_binary_values(SOUR + ':DATA:ARB '+name+',', sig, datatype='f', is_big_endian=False)

    #wait until that command is done
    inst.write('*WAI')

    #name the arb
    inst.write(SOUR + ':FUNC:ARB '+name)

    #set sample rate, voltage
    inst.write(SOUR + ':FUNC:ARB:SRAT ' + sRate)
    inst.write(SOUR + ':VOLT:OFFS 0')
    inst.write(SOUR + ':FUNC ARB')
    inst.write(SOUR + ':VOLT '+args.pulseheight)
    inst.write('FUNC:ARB:SYNC')  # Chen: this sync arbs is important to align two chrips with internal trigger.  equivalen to Parameters/more/Sync Arbs

    #save arb to device internal memory
    #inst.write('MMEM:STOR:DATA "INT:\\remoteAdded\\'+name+'.arb"') # Chen: No need to save to intermal memory, I can update it.

    #clear message
    inst.write("DISP:TEXT ''")

    #check for error messages
    instrument_err = "error"
    while instrument_err != '+0,"No error"\n':
        inst.write('SYST:ERR?')
        instrument_err = inst.read()
        if instrument_err[:4] == "-257":  #directory exists message, don't display
            continue;
        if instrument_err[:2] == "+0":    #no error
            continue;
        print(instrument_err)


    #close device
    inst.close()

    #generate a macro to load this arb using AgilentControl.py
    if args.macro:
        macroFile = "load_"+name+".awg"
        with open(macroFile, 'w') as f:
            f.write("# Macro generated by UploadArb.py \n")
            f.write("MMEMORY:LOAD:DATA1 \"INT:\\remoteAdded\\"+name+".arb\"\n")
            f.write("SOURCE"+ SOUR[-1] + ":FUNCTION ARB\n")
            f.write("SOURCE" + SOUR[-1] + ":FUNCtion:ARBitrary \"INT:\\remoteAdded\\"+name+".arb\"\n")


if __name__ == '__main__':
    # x_ch1 should be real-valued signal for the equalization.
    x_ch1 =  coe.y_cx.real #np.load(setup.file_tx)# 0.0001*np.ones([coe.N, 1]) #coe.y_cx.real #0.0001*np.ones([coe.N, 1]) # coe.y_cx.real #0.001*np.ones([coe.N, 1]) #coe.y_cx.real #0.001*np.ones([coe.N, 1])#coe.y_cx.real #np.load('x_canc_PA.npy') #readcsv('output_cal_antenna_ch2_EQ.csv') #readcsv('output_cal_Mixer_ch2_after_EQ.csv') # ideal copy: use the channel 1 signal as tx template
    x_ch1 = x_ch1 / max(abs(x_ch1))
    step = 2


    SOUR = 'SOUR1'
    if step == 1: # When operate this step, turn off channel 2 of 33600A
        # change readosc scope.waveform(fn, '1', itt, points=500000)  # use this one for long chirp 20 ms
        # or change to scope.waveform(fn, '1', itt, points=4000)  for 1.6 us chirp
        UploadArb(x_ch1,SOUR = SOUR)  # step 1
        readosc.readosc(filename='data/'+setup.EQ_filename) # step 1 Turn off the other channel.
        #readosc.readosc(filename='data/rubish.csv')
    else: # step = 2
        y = readosc.readcsv(filename='data/'+setup.EQ_filename)  # step 2: a pre-record loopback waveform.
        x_ch1_EQ = equalizer(x_ch1, y, input=x_ch1, scale=0) # step 2 with normalized scale
        UploadArb(x_ch1_EQ,SOUR = SOUR ) # step 2

    ##################
    # Original EQ
    ##################
    '''
    x = coe.y_cx.real
    x_EQ = equalizer(x, y, input = x)
    UploadArb(x_EQ)
    '''
    #n = np.linspace(0, len(x), len(x))
    #plt.plot(n, x, '-.', label='Original Signal')
    #plt.plot(n, x_EQ, '*-', label='With Equalizer')
    #plt.legend()
    #plt.show()