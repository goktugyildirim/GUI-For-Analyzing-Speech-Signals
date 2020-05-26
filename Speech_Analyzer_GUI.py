#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Fri May  26 18:50:54 2020

@author: Göktuğ YILDIRIM 
"""

from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
#from playsound import playsound
#import wavio
from scipy.signal import find_peaks
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

import sys
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random

import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QToolTip, QMessageBox, QLabel)


        
        

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Speech Analyzer"
        self.setWindowTitle(self.title)
        #self.setGeometry(575,100,800,800)
        #self.setFixedSize(640, 480)


        widget = QWidget()
        v_box = QVBoxLayout()
        h1_box = QHBoxLayout()
        h2_box = QHBoxLayout()
        h3_box = QHBoxLayout()
        h4_box = QHBoxLayout()
        h5_box = QHBoxLayout()
        
        h10_box = QHBoxLayout()
        
        window_length_text = QLabel()
        window_length_text.setText("Window duration in second:")
        self.win_dur = QLineEdit()
        self.win_dur.setPlaceholderText("second")
        h2_box.addWidget(window_length_text)
        h2_box.addWidget(self.win_dur)
        v_box.addLayout(h2_box)
        
        
        frame_shift_text = QLabel()
        frame_shift_text.setText("Frame shift in second:")
        self.frame_shift = QLineEdit()
        self.frame_shift.setPlaceholderText("second")
        self.N = QLineEdit()
        self.N.setPlaceholderText("N")
        h2_box.addWidget(frame_shift_text)
        h2_box.addWidget(self.frame_shift)
        h2_box.addWidget(self.N)
        v_box.addLayout(h2_box)
        
        
        
        start_id_text = QLabel()
        start_id_text.setText("Start id:")
        self.start_id = QLineEdit()
        self.start_id.setPlaceholderText("sample")
        h3_box.addWidget(start_id_text)
        h3_box.addWidget(self.start_id)
        
        end_id_text = QLabel()
        end_id_text.setText("End id:")
        self.end_id = QLineEdit()
        self.end_id.setPlaceholderText("sample")
        h3_box.addWidget(end_id_text)
        h3_box.addWidget(self.end_id)
        
        button3 = QPushButton("Cut")
        button3.clicked.connect(self.cut)
        h3_box.addWidget(button3)
        
        
        self.path = QLineEdit()
        self.path.setPlaceholderText("C://Windows")
        
        button1 = QPushButton("Read Sound")
        button1.clicked.connect(self.readSound)
        
    
        
        h1_box.addWidget(self.path)                         # h2box h1box information button2(plot-part)
        h1_box.addWidget(button1)

        self.information = QLabel()
        
        
        
        plot_waveform = QPushButton("Plot Waveform")
        plot_waveform.clicked.connect(self.plotWaveform)
        plot_fft = QPushButton("Plot FFT")
        plot_fft.clicked.connect(self.plotFFT)
        plot_energy = QPushButton("Plot Short-Time Log Energy")
        plot_energy.clicked.connect(self.plotEnergy)
        plot_zc = QPushButton("Plot Short-Time Zero Crossing")
        plot_zc.clicked.connect(self.plotZC)
        plot_spect = QPushButton("Plot Spectrogram")
        plot_spect.clicked.connect(self.plotSpect)
        plot_formants = QPushButton("Extract Formants")
        plot_formants.clicked.connect(self.plotFormants)
        plot_pitch = QPushButton("Extract Pitch")
        plot_pitch.clicked.connect(self.plotPitch2)
        plot_pause = QPushButton("Pause Durations")
        plot_pause.clicked.connect(self.pauseDuration)
        
        

        
        #(self.N.text())
        
        
        h4_box.addWidget(plot_waveform)
        h4_box.addWidget(plot_fft)
        h4_box.addWidget(plot_energy)
        h4_box.addWidget(plot_zc)
        h4_box.addWidget(plot_spect)
        h4_box.addWidget(plot_formants)
        h4_box.addWidget(plot_pitch)
        h4_box.addWidget(plot_pause)
        
        self.frame = QLineEdit()
        self.frame.setPlaceholderText("frame")
        plot_STFT = QPushButton("Plot Short-Time Fouirer Transform")
        plot_STFT.clicked.connect(self.plotSTFT)
        plot_cepstrum = QPushButton("Plot Short-Time Real Cepstrum")
        plot_cepstrum.clicked.connect(self.plotCepstrum)
        h5_box.addWidget(self.frame)
        h5_box.addWidget(plot_STFT)
        h5_box.addWidget(plot_cepstrum)
        
        
        
        v_box.addLayout(h1_box)
        v_box.addWidget(self.information)
        v_box.addLayout(h3_box)
        v_box.addLayout(h4_box)
        v_box.addLayout(h5_box)
        
        names = QLabel()
        names.setText("Göktuğ YILDIRIM")
        h10_box.addWidget(names)
        
        
        
       

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QHBoxLayout()
        layout.addWidget(self.toolbar)
        
        v_box.addLayout(layout)
        v_box.addWidget(self.canvas)
        v_box.addLayout(h10_box)
        
        
             
        
        
        widget.setLayout(v_box)
        self.setCentralWidget(widget)
        self.show()

        
        


            
            
        
    def plotWaveform(self):  
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.data, '*-')
        self.canvas.draw()    
        #self.figure.clear()
    
    def plotFFT(self):   
        x = float(self.N.text())
        N = int(x)
        import numpy as np
        #N = int(self.N.text())
        dft_frame = np.fft.fft(self.data,N)
        self.figure.clear()
        f = np.arange(0,self.fs/2,self.fs/N)
        ax = self.figure.add_subplot(111)
        ax.plot(f,np.abs(dft_frame[:int(len(dft_frame)/2)])/max(dft_frame))
        #plt.xlabel("Hz")
        self.canvas.draw()       
        #self.figure.clear()
    
    
    def plotEnergy(self):
        import numpy as np
        energy_Vector = []
        sum = 0
        for frame in (self.windowed_data):
            sum = 0
            for i in range(len(frame)):
                sum = sum + frame[i]*frame[i]
            energy_Vector.append(sum)
        
        energy_Vector = np.array(energy_Vector).reshape((len(self.windowed_data),1))
        energy_Vector = energy_Vector/max(energy_Vector)
        energy_Vector = 10*np.log10(energy_Vector)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(energy_Vector, '*-')
        #plt.xlabel("Frames")
        self.canvas.draw()  
        
        
        
    def plotZC(self):
        
        shift_length = int(self.fs*float(self.frame_shift.text()))
        
        import numpy as np
        zc_vector = []
        
        for frame in (self.windowed_data):
            sum = 0
            for i in range(len(frame)-1):
                first_element = frame[i]
                second_element = frame[i+1]
                element = np.abs(np.sign(first_element)-np.sign(second_element))
                sum = sum + element
            zc = (shift_length*sum)/(len(frame)*2)
            zc_vector.append(zc)#her window için tek bir zc değeri
    
        zc_vector = np.array(zc_vector).reshape((len(self.windowed_data),1))
        zc_vector = zc_vector/max(zc_vector)
        zc_vector = 10*np.log10(zc_vector)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(zc_vector, '*-')
        #plt.xlabel("Frames")
        #plt.xlabel("Frames")
        self.canvas.draw()     
        

    def plotSTFT(self):
        x = float(self.N.text())
        N = int(x)
        import numpy as np
        
        STFT = []
        dft_frame = []
        
        for i,frame in enumerate(self.windowed_data):            
            
            dft_frame = np.fft.fft(frame,N)
            STFT.append(dft_frame)
            
    
        STFT = np.array(STFT)
        x = float(self.frame.text())
        frame = int(x)
        dft_frame = STFT[frame]
        self.figure.clear()
        f = np.arange(0,self.fs/2,self.fs/N)
        ax = self.figure.add_subplot(111)
        ax.plot(f,np.abs(dft_frame[:int(len(dft_frame)/2)]))
        #plt.xlabel("Hz")
        self.canvas.draw()       
             
    
    def plotSpect(self):
        x = float(self.N.text())
        N = int(x)
        import numpy as np
        
        STFT = []
        dft_frame = []
        
        for i,frame in enumerate(self.windowed_data):            
            
            dft_frame = np.fft.fft(frame,N)
            STFT.append(dft_frame)
            
        STFT = np.array(STFT)
        
        import numpy as np
        from scipy import signal
        import matplotlib.pyplot as plt
        
        
        
        STFT = np.transpose(STFT)
        STFT = STFT[:int(N/2)][:]
        f = np.arange(0,self.fs/2,self.fs/N)
        f = f.reshape((f.shape[0],1))
        t = np.arange(0,STFT.shape[1],1)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pcolormesh(t, f, 20*np.log10(np.abs(STFT)))
        self.canvas.draw() 
                 

            
    def plotCepstrum(self):
        x = float(self.N.text())
        N = int(x)
        
        import numpy as np
        
        cepstrum = []
                
        for i,frame in enumerate(self.windowed_data):     
            
            dft_frame = np.fft.fft(frame,N)
            
            ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
            
            
            """if lp_liftering == True:
                ones = np.ones((1,cutoff))
                zeros = np.zeros((1,(1024-(2*cutoff))))
                lif = np.concatenate((ones,zeros,ones),axis=1).ravel()
                ceps_frame = ceps_frame*lif"""
            
            cepstrum.append(ceps_frame)
            
        x = float(self.frame.text())
        frame = int(x)
        ceps_frame = cepstrum[frame] 
        ceps_frame = ceps_frame/max(ceps_frame)
        f = np.arange(0,N,1)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(ceps_frame) 
        #scale_factor = 0.05
        #ymin, ymax = plt.ylim()
        #ax.ylim(ymin * scale_factor, ymax * scale_factor)
        self.canvas.draw() 
        
        
        #plt.grid(True)
        #plt.title("Cepstrum of Frame: {}: ".format(i))
        #plt.xlabel("Quefrency (Sample Id)")
        #plt.figure()
            
            
    def plotFormants(self):
        x = float(self.N.text())
        N = int(x)
        
        import numpy as np
        
        cepstrum = []
                
        for i,frame in enumerate(self.windowed_data):     
            
            dft_frame = np.fft.fft(frame,N)
            
            ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
            
            total = []
            total = np.array(total)
            
            if True:
                ones = np.ones((1,15))
                zeros = np.zeros((1,(1024-(2*15))))
                lif = np.concatenate((ones,zeros,ones),axis=1).ravel()
                ceps_frame = ceps_frame*lif
                dft_frame = np.fft.fft(ceps_frame,N)
                dft_frame = abs(dft_frame)
                
                
                if i in (self.list_v):
                    aranan = 0
                    for id, value in enumerate(self.list_v):
                        if value==i:
                            aranan = id
                    total = self.voiced_peaks[aranan]
                    cepstrum.append(total)
                 
                    
                if i in (self.list_u):
                    aranan = 0
                    for id, value in enumerate(self.list_u):
                        if value==i:
                            aranan = id
                    total = self.unvoiced_peaks[aranan]
                    cepstrum.append(total)                   
                    
                else:
                    cepstrum.append(dft_frame)
         
                
            
        cesptrum = np.array(cepstrum)
        STFT = cepstrum
        
        #print("STFT shape",STFT.shape)
        
        #import numpy as np
        #from scipy import signal
        #import matplotlib.pyplot as plt
        
        
        STFT = np.transpose(STFT)
        STFT = STFT[:int(N/2)][:]
        f = np.arange(0,self.fs/2,self.fs/N)
        f = f.reshape((f.shape[0],1))
        t = np.arange(0,STFT.shape[1],1)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pcolormesh(t, f, np.abs(STFT))
        #ax.xlabel("Frames")
        self.canvas.draw() 
        
        

            
            
            
        
    
    def readSound(self):
        import numpy as np
        self.fs, self.data = wavfile.read(self.path.text())
        b = signal.firwin(101, cutoff=100, fs=self.fs, pass_zero=False)
        self.data = signal.lfilter(b, [1.0], self.data)
        self.fs = int(self.fs)
        self.windowed_data = self.window(self.data, self.fs, float(self.win_dur.text()), float(self.frame_shift.text()))
        self.information.setText("Sound length: {}\nTotal frame count: {} \nSampling Rate: {} Hz\nWindow Length: {}".format(self.data.shape[0],len(self.windowed_data), str(self.fs), len(self.windowed_data[0])))
        
        self.list_v1, self.list_u, list_s = self.voicedUnvoicedClassification(self.data, self.fs, float(self.win_dur.text()), float(self.frame_shift.text()))
        
        self.list_v = self.pad(self.list_v1, 25)
        self.list_u = self.pad(self.list_u, 25)
        
        
        #self.pauseDuration()
        
        
        self.voiced_peaks = self.collect_peaks(self.list_v, self.windowed_data, 20)
        
        self.unvoiced_peaks = self.collect_peaks(self.list_u, self.windowed_data, 20)

        
        
        
    def pauseDuration(self):
        setV = list(set(self.list_v))
        setU = list(set(self.list_u))
        no_silence = setV+setU
        no_silence = set(no_silence)
        no_silence = list(no_silence)
        #print("Ses olan idler",no_silence)
        
        import numpy as np
        x = float(self.N.text())
        N = int(x)
        
        plot_list = []
        
        for i,frame in enumerate(self.windowed_data):
            zeros = np.zeros((1,N)).ravel()
            if i in no_silence:
                zeros[int(N/4)] = 10000
                plot_list.append(zeros)
                
            else:
                #print("ses yok",i)
                zeros[int(N/4)] = 10000000
                plot_list.append(zeros) 
                
        plot_list = np.array(plot_list)
        
        
        STFT = np.log10(plot_list)
        
        #import numpy as np
        #from scipy import signal
        #import matplotlib.pyplot as plt
        
        
        STFT = np.transpose(STFT)
        STFT = STFT[:int(N/2)][:]
        f = np.arange(0,self.fs/2,self.fs/N)
        f = f.reshape((f.shape[0],1))
        t = np.arange(0,STFT.shape[1],1)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pcolormesh(t, f, np.abs(STFT))
        #ax.xlabel("Frames")
        self.canvas.draw()         
                
        
        
        
        
        
        
        

        
        
        
        
        
        
    def cut(self):
        self.data = self.data[int(self.start_id.text()):int(self.end_id.text())]
        self.windowed_data = self.window(self.data, self.fs, float(self.win_dur.text()), float(self.frame_shift.text()))
        self.information.setText("Sound length: {}\nTotal frame count: {} \nSampling Rate: {} Hz\nWindow Length: {}".format(self.data.shape[0],len(self.windowed_data), str(self.fs), len(self.windowed_data[0])))
        
        self.list_v, self.list_u, list_s = self.voicedUnvoicedClassification(self.data, self.fs, float(self.win_dur.text()), float(self.frame_shift.text()))
        
        #print("Before padding:",len(self.list_v),len(self.list_u))
        
        self.list_v = self.pad(self.list_v, 25)
        self.list_u = self.pad(self.list_u, 25)
        #print(self.list_v, self.list_u)
        #print("After padding:",len(self.list_v),len(self.list_u))
        #print("Voiced frames:")
        self.voiced_peaks = self.collect_peaks(self.list_v, self.windowed_data, 20)
        #print("Unvoiced frames:")
        self.unvoiced_peaks = self.collect_peaks(self.list_u, self.windowed_data, 20)

        #print("After padding:",len(self.voiced_peaks),len(self.unvoiced_peaks))
        
            

    def window2(self):    
        pass
        #self.w = Window2(self.data, self.waveform.isChecked, self.fft.isChecked, self.N.text(), self.fs)
        #self.w.show()


        
        
    #**************************************************
    
    def plotPitch(self):
        import numpy as np
        x = float(self.N.text())
        N = int(x)
        
        pitch_freq_male_max = 260
        pitch_freq_female_max = 525

        pitch_per_male_max = 1/pitch_freq_male_max
        pitch_per_female_max = 1/pitch_freq_female_max

        id1 = int(self.fs*pitch_per_male_max)
        id2 = int(self.fs*pitch_per_female_max)

        min_id = min(id1,id2)
        max_id = int(1024/2)
        
        
        pitch_id_list = []
        
        for i,frame in enumerate(self.windowed_data):
            
            if i in self.list_v:
                ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
                
                max_amplitude = 0
                aranan_id = 0

                for i in range(max_id):
                    if i>min_id:
                        if ceps_frame[i]>max_amplitude:
                            max_amplitude = ceps_frame[i]
                            aranan_id = i

                pitch_per = aranan_id /self.fs
                pitch_freq = 1/pitch_per
                pitch_freq_id = int(pitch_freq / (self.fs/N))
                pitch_id_list.append(pitch_freq_id)
                
                
        pitch_id_list = self.makeSmoothPitch(pitch_id_list,15)
        show = []
        k = 0
        for i,frame in enumerate(self.windowed_data):
            
            if i in self.list_v:

                dft_frame = abs(np.fft.fft(frame,N))
        
                zeros = np.zeros((1,1024)).ravel()
                pitch_freq_id = pitch_id_list[k]
                k = k + 1
                zeros[pitch_freq_id] = 1000000
                total = zeros + dft_frame

                show.append(total)
                
            else:
                dft_frame = np.ones((1,1024)).ravel()
                show.append(dft_frame)





        
        show = np.array(show)
        STFT = show
        
        #import numpy as np
        #from scipy import signal
        #import matplotlib.pyplot as plt
        
        
        STFT = np.transpose(STFT)
        STFT = STFT[:int(N/2)][:]
        f = np.arange(0,self.fs/2,self.fs/N)
        f = f.reshape((f.shape[0],1))
        t = np.arange(0,STFT.shape[1],1)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pcolormesh(t, f, np.abs(STFT))
        #ax.xlabel("Frames")
        self.canvas.draw()         
               


        #average_pitch_per_of_voiced_frames = np.mean(pitch_per_list)   
        #average_pitch_freq_of_voiced_frames = 1 / average_pitch_per_of_voiced_frames
    #*****************************************************  
        
        
        
    def plotPitch2(self):
        import numpy as np
        x = float(self.N.text())
        N = int(x)
        
        pitch_freq_male_max = 260
        pitch_freq_female_max = 525

        pitch_per_male_max = 1/pitch_freq_male_max
        pitch_per_female_max = 1/pitch_freq_female_max

        id1 = int(self.fs*pitch_per_male_max)
        id2 = int(self.fs*pitch_per_female_max)

        min_id = min(id1,id2)
        max_id = int(N/2)
        
        
        pitch_freq_list = []
        
        for i,frame in enumerate(self.windowed_data):
            
            if i in self.list_v1:
                ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
                
                max_amplitude = 0
                aranan_id = 0

                for i in range(max_id):
                    if i>min_id:
                        if ceps_frame[i]>max_amplitude:
                            max_amplitude = ceps_frame[i]
                            aranan_id = i

                pitch_per = aranan_id /self.fs
                pitch_freq = 1/pitch_per
                pitch_freq_list.append(pitch_freq)
                
        
        pitch_freq_list = self.makeSmoothPitch(pitch_freq_list,30)#smoothing
        
        k = 0
        total_freq = []
        for i,frame in enumerate(self.windowed_data):   
            
            if i in self.list_v1:
                total_freq.append(pitch_freq_list[k])
                k = k + 1
            else:
                total_freq.append(0)
                
                
        

        
        window_number=len(self.windowed_data)
        x = np.arange(window_number)
        y = total_freq
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = (0,0,0)
        ax.scatter(x, y, s=5, c=colors, alpha=0.9)
        self.canvas.draw()


        

   
               


        #average_pitch_per_of_voiced_frames = np.mean(pitch_per_list)   
        #average_pitch_freq_of_voiced_frames = 1 / average_pitch_per_of_voiced_frames
    #*****************************************************         
        
        
        
        
        
        



    def voicedUnvoicedClassification(self,data, fs, window_dur_in_second, frame_shift ):
        import numpy as np
        windowed_data = self.window(data,fs, window_dur_in_second = window_dur_in_second, frame_shift = frame_shift)
        
        output = {}
        energy = self.shortTimeEnergy(windowed_data)
        zcr = self.shortTimeZeroCrossing(windowed_data, frame_shift = frame_shift, fs = fs)
        average_energy = np.mean(energy)
        average_zcr = np.mean(zcr)
        
        list_v = []
        list_u = []
        list_s = []   
            
        for i, energy_frame in enumerate(energy): #ilk kriter for voiced
            if energy_frame > average_energy and energy_frame>zcr[i]:
                list_v.append(i)
            else:#ilk kriter for unvoiced
                if energy_frame<max(average_energy,zcr[i]) and (zcr[i]>average_zcr*1.5 or energy_frame+zcr[i]<average_energy):
                    list_u.append(i)
                else:
                    list_s.append(i)
        
        return list_v, list_u, list_s
    

    
    def window(self, x, fs, window_dur_in_second, frame_shift):
        import numpy as np
        sound_length = len(x)
        window_length = int(fs*window_dur_in_second)
        shift_length = int(fs*frame_shift)
        num_window = int((sound_length-window_length)/shift_length + 1)

        #print("Sound length", sound_length)
        #print("Sampling Rate:", fs)
        #print("Window length", window_length)
        #print("Shift length:",shift_length)
        #print("Num window: ",num_window)

        windowed_data = []

        for i in range(int(num_window)):
            window = [0.54-0.46*np.cos((2*3.14*i)/(window_length-1)) for i in range(window_length)]
            frame = x[(i*shift_length):(i*shift_length)+window_length]*window
            windowed_data.append(frame)

        return windowed_data
    
    

    def shortTimeZeroCrossing(self, windowed_data, frame_shift, fs):
        
        shift_length = int(fs*frame_shift)
        
        import numpy as np
        zc_vector = []
        
        for frame in (windowed_data):
            sum = 0
            for i in range(len(frame)-1):
                first_element = frame[i]
                second_element = frame[i+1]
                element = np.abs(np.sign(first_element)-np.sign(second_element))
                sum = sum + element
            zc = (shift_length*sum)/(len(frame)*2)
            zc_vector.append(zc)#her window için tek bir zc değeri
    
        zc_vector = np.array(zc_vector).reshape((len(windowed_data),1))
        zc_vector = zc_vector/max(zc_vector)
        #zc_vector = 10*np.log10(zc_vector)
        
        return zc_vector
    
    
    
    def shortTimeEnergy(self, windowed_data):
        import numpy as np
        energy_Vector = []
        sum = 0
        for frame in (windowed_data):
            sum = 0
            for i in range(len(frame)):
                sum = sum + frame[i]*frame[i]
            energy_Vector.append(sum)
        
        energy_Vector = np.array(energy_Vector).reshape((len(windowed_data),1))
        energy_Vector = energy_Vector/max(energy_Vector)
        #energy_Vector = 10*np.log10(energy_Vector)
        
        return energy_Vector
    
    
    def pad(self,voiced,n):
        length = len(voiced)

        empty = []

        for i in range(length-1):

            difference = voiced[i+1]-voiced[i]

            if difference<n:

                if difference>1:#padding

                    empty.append(voiced[i])

                    for k in range(difference-1):
                        element = voiced[i] + k + 1
                        empty.append(element)

                if difference==1:#direk al
                    empty.append(voiced[i])

        return empty        
 
    def collect_peaks(self, voiced_list, windowed_data, window_length):
        import numpy as np
        x = float(self.N.text())
        N = int(x)

        
        all_peaks = []
        peak_frames = []
        for  i, frame in enumerate(windowed_data):

            if i in (voiced_list):
                dft_frame = np.fft.fft(frame,N)
                ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
                total = []
                total = np.array(total)

                ones = np.ones((1,15))
                zeros = np.zeros((1,(1024-(2*15))))
                lif = np.concatenate((ones,zeros,ones),axis=1).ravel()
                ceps_frame = ceps_frame*lif
                dft_frame = np.fft.fft(ceps_frame,N)
                dft_frame = abs(dft_frame)

                peaks, _ = find_peaks(dft_frame[:int(len(dft_frame)/2)], height=0)
                all_peaks.append(peaks)
                """zeros = np.zeros((1,1024)).ravel()
                for peak in list(peaks):
                    zeros[peak]=10000
                total = zeros + dft_frame
                peak_frames.append(total)"""

        smooth_peaks = self.makeSmooth(all_peaks, window_length)    
        #print("smooth_peaks",smooth_peaks)
        
        peak_frames = []
        k = 0
        for  i, frame in enumerate(windowed_data):
            
            if i in (voiced_list):
                #print("frame:",i)
                dft_frame = np.fft.fft(frame,N)
                ceps_frame = np.real(np.fft.ifft(np.log10(np.abs(np.fft.fft(frame,N))))).reshape((N,1)).ravel()
                total = []
                total = np.array(total)

                ones = np.ones((1,15))
                zeros = np.zeros((1,(1024-(2*15))))
                lif = np.concatenate((ones,zeros,ones),axis=1).ravel()
                ceps_frame = ceps_frame*lif
                dft_frame = np.fft.fft(ceps_frame,N)
                dft_frame = abs(dft_frame)

                peaks = smooth_peaks[k]
                #print("Peaks:", peaks)
                k = k + 1
                
                zeros = np.zeros((1,1024)).ravel()
                for peak in list(peaks):
                    zeros[peak] = 1000000
                total = zeros + dft_frame
                peak_frames.append(total)     
                
                
                
        return peak_frames
    
    
    
    
    
    
    
    
    
    
    def makeSmooth(self, peaks, window_length):
        
        import numpy as np
        sum_f1 = 0
        sum_f2 = 0
        sum_f3 = 0
        sum_f4 = 0
        sum_f5 = 0

        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0

        for frame in peaks:
            if len(frame)>=1:
                sum_f1 = sum_f1 + frame[0]
                c1 += 1
            if len(frame)>=2:
                sum_f2 = sum_f2 + frame[1]
                c2 += 1
            if len(frame)>=3:
                sum_f3 = sum_f3 + frame[2]
                c3 += 1
            if len(frame)>=4:
                sum_f4 = sum_f4 + frame[3]
                c4 += 1


        av_f1 = int(sum_f1/c1)
        av_f2 = int(sum_f2/c2)
        av_f3 = int(sum_f3/c3)
        av_f4 = int(sum_f4/c4)


        last_peak = []

        for frame in peaks:
            empty = []
            if len(frame)>4:
                for i in range(4):
                    empty.append(frame[i])
                last_peak.append(empty)

            else:
                last_peak.append(frame)

        #print(av_f1,av_f2,av_f3,av_f4)
        #print(last_peak)

        last_peak2 = []

        for frame in last_peak:
            empty = []
            if len(frame)==4:
                last_peak2.append(frame)
            else:
                empty.append(frame[0])
                empty.append(frame[1])
                empty.append(av_f3)
                empty.append(av_f4)
                last_peak2.append(empty)


        last_peak3 = []

        #window_length = 2

        for i in range(len(last_peak2)-(window_length-1)):
            empty = []
            window = last_peak2[i:i+window_length]
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0

            for frame in window:
                f1 = f1 + frame[0]
                f2 = f2 + frame[1]
                f3 = f3 + frame[2]
                f4 = f4 + frame[3]

            empty = [int(f1/window_length),int(f2/window_length),int(f3/window_length),int(f4/window_length)]
            last_peak3.append(empty)

        length_peaks = len(peaks)
        length_last_peak3 = len(last_peak3)
        difference = length_peaks - length_last_peak3
        last_element = last_peak3[-1]

        for i in range(difference):
            last_peak3.append(last_element)
        
        return last_peak3
    
    
    
    def makeSmoothPitch(self, pitch_id, window_length):
        import numpy as np
        #print(len(pitch_id))

        last_pitch_ids = []

        for i in range(len(pitch_id)-(window_length-1)):
            mean = int(np.mean(pitch_id[i:i+window_length]))
            last_pitch_ids.append(mean)

        last_element = mean

        for i in range(len(pitch_id)-len(last_pitch_ids)):
            last_pitch_ids.append(last_element)

        return last_pitch_ids
    
    
    
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())


# In[ ]:





# In[4]:





# In[36]:



    


# In[3]:





# In[ ]:





# In[ ]:




