import numpy as np
import soundfile as sf

def sisdr(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    """
    # Align the signals in time
    if len(reference) > len(estimation):
        reference = reference[:len(estimation)]
    else:
        estimation = estimation[:len(reference)]
    #estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return np.mean(10 * np.log10(ratio))

def gpt_sisdr(reference, estimation):
    """
    Calculate the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    :param reference: The reference audio signal (clean, original).
    :param estimation: The estimated audio signal (processed).
    :return: The SI-SDR value.
    """
    # Ensure the signals are numpy arrays
    reference = np.asarray(reference)
    estimation = np.asarray(estimation)

    # Align the signals in time
    if len(reference) > len(estimation):
        reference = reference[:len(estimation)]
    else:
        estimation = estimation[:len(reference)]

    # Scale the estimated signal to have the same energy as the reference signal
    estimation *= np.sqrt(np.sum(reference ** 2) / np.sum(estimation ** 2))

    # Calculate the signal-to-noise ratio
    noise = estimation - reference
    sisdr = 10 * np.log10(np.sum(reference ** 2) / np.sum(noise ** 2))

    return sisdr

# Load the reference and estimation WAV files
r, _ = sf.read('/n/work1/shi/corpus/wsj/4a2c0405.wav')

print('Source 1')
# Calculate the SI-SDR
for i in range(4):
    e, _ = sf.read('/n/work1/juchen/SoundSourceSeparation/src_torch/separation/ilrma_wpe_source{}.wav'.format(i))
    sdr1 = sisdr(r, e)
    sdr2 = gpt_sisdr(r, e)
    print(sdr1, sdr2)

r, _ = sf.read('/n/work1/shi/corpus/wsj/48lc041c.wav')
print('Source 2')
# Calculate the SI-SDR
for i in range(4):
    e, _ = sf.read('/n/work1/juchen/SoundSourceSeparation/src_torch/separation/ilrma_wpe_source{}.wav'.format(i))
    sdr1 = sisdr(r, e)
    sdr2 = gpt_sisdr(r, e)
    print(sdr1, sdr2)