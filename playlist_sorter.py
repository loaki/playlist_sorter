import argparse
import array
import math
import wave
import sys
import os
import matplotlib.pyplot as plt
import numpy
import pywt
from os import path
from pydub import AudioSegment
import shutil
import ffmpeg
from scipy import signal


def read_wav(filename):
    # open file, get metadata for audio
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    # Read entire file and make into an array
    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


# print an error when no data can be found
def no_audio_data():
    #print("No audio data for sample, skipping...")
    return None, None


# simple peak detection
def peak_detect(data):
    max_val = numpy.amax(abs(data))
    peak_ndx = numpy.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = numpy.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = numpy.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - numpy.mean(cD)

        # 6) Recombine the signal before ACF
        #    Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0 : math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - numpy.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = numpy.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        return no_audio_data()

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    #print(bpm)
    return bpm, correl

def get_bpm(filename, window):
    samps, fs = read_wav(filename)
    data = []
    correl = []
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(window * fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = numpy.zeros(max_window_ndx)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        # print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
        data = samps[samps_ndx : samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        bpms[window_ndx] = bpm
        correl = correl_temp

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    bpm = numpy.median(bpms)
    #print("Completed.  Estimated Beats Per Minute:", bpm)
    return bpm

def sort(files_in_dir, files_bpm):
    list_len = len(files_in_dir)
    for rank in range(0, list_len):
        low_bpm = files_bpm[0]
        position = 0
        i = 0
        for item in files_in_dir:
            if files_bpm[i] < low_bpm:
                low_bpm = files_bpm[i]
                position = i
            i += 1
        if os.path.isfile('input/'+files_in_dir[position].replace('.wav', '.mp3')):
            shutil.copy('input/'+files_in_dir[position].replace('.wav', '.mp3'), 'sorted/'+str(rank)+'_'+files_in_dir[position].replace('.wav', '.mp3'))
        else:
            shutil.copy('input/'+files_in_dir[position], 'sorted/'+str(rank)+'_'+files_in_dir[position])
        files_in_dir.remove(files_in_dir[position])
    print(list_len,'files sorted')

def conv_wav(location, files_conv):
    for r, d, f in os.walk(location):
        for item in f:
            if '.mp3' in item:
                #print('input/'+item,'input/'+item.replace('.mp3', '.wav'))
                sound = AudioSegment.from_mp3('input/'+item)
                sound.export('input/'+item.replace('.mp3', '.wav'), format="wav")
                files_conv.append(item.replace('.mp3', '.wav'))

if __name__ == "__main__":
    location = os.getcwd()+'\\input'
    files_in_dir = []
    files_bpm = []
    files_conv = []
    #print(location)
    conv_wav(location, files_conv)
    # r=>root, d=>directories, f=>files
    for r, d, f in os.walk(location):
        for item in f:
            if '.wav' in item:
                files_in_dir.append(item)
    i = 0
    for item in files_in_dir:
        bpm = get_bpm('input/'+item, 5)
        files_bpm.append(bpm)
        print(item, ' : ',int(bpm),'bpm')
        i += 1
    sort(files_in_dir, files_bpm)
    for item in files_conv:
        os.remove('input/'+item)