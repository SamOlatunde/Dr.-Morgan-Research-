#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
fs_rate, signal = wavfile.read(r"C:\Users\User\Repositories\Research\Dr.-Morgan-Research-\similar songs\chrome slugs\03-thuggish-ruggish-bone.wav")
print ("Frequency sampling", fs_rate)
l_audio = len(signal.shape)
#with open("signal.txt","w") as file:
"""signalOut =  np.array(signal)
file = open("Signal.txt", "w")
for row in signal:
    file.write("{a}  {b}\n".format(a=row[0], b = row[1]))"""
print ("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
print ("Complete Samplings N", N)
secs = N / float(fs_rate)
print ("secs", secs)
Ts = 1.0/fs_rate # sampling interval in time
print ("Timestep between samples Ts", Ts)
t = np.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
#for i in range(len(t)):
 #   print(t)
FFT = abs(scipy.fft.fft(signal))
FFT_side = FFT[range(N//2)] # one side FFT range
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] # one side frequency range
fft_freqs_side = np.array(freqs_side)
plt.subplot(311)
"""p1 = plt.plot(t, signal, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(312)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum"""
#my addition - plotting frequency against time
p2 = plt.plot(t, freqs,'y')
"""plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.subplot(313)
p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')"""
plt.show()
