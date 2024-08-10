import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft

path = ['/n/work1/shi/corpus/wsj/48ic0208.wav','/n/work1/juchen/BSS/scripts/bss_output/19_48ic0208_fastmnmf.wav','/n/work1/juchen/BSS/scripts/bss_output/19_48ic0208_fastmnmf+mvdr(wtsum).wav','/n/work1/juchen/BSS/recordings_noisy_noisy/48ic0208.49gc040v.wav']

wave_data = []
for file in path:
    sr, wav = wavfile.read(file)
    print(sr, wav.shape)
    wave_data.append(wav)
# print(wave_data[0].shape)
# print(sr)

plt.figure(figsize=(12, 8))
# show
plt.subplot(2, 2, 1)
plt.specgram(wave_data[0], Fs=sr, scale_by_freq=True, sides = 'default')
# plt.colorbar(format='%+2.0f dB')
# plt.title("Mixture")

plt.subplot(2, 2, 2)
plt.specgram(wave_data[1], Fs=sr, scale_by_freq=True, sides = 'default')
# plt.colorbar(format='%+2.0f dB')
# plt.title("Source 1")

plt.subplot(2, 2, 3)
plt.specgram(wave_data[2], Fs=sr, scale_by_freq=True, sides = 'default')
# plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 2, 4)
plt.specgram(wave_data[3][:, 1], Fs=sr, scale_by_freq=True, sides = 'default')
# plt.colorbar(format='%+2.0f dB')

# plt.title("Source 2")
plt.show()
plt.savefig("spectrogram.png")