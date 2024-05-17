import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import torch
import torchaudio
import numpy as np
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile

import os, random
dir = "/n/work1/juchen/BSS/recordings/"
wav_files = os.listdir(dir)
wav_files = random.sample(wav_files, 20)
# wav_files[0] = "46jc0311.489c0310.wav"

# ref_file = "/n/work1/juchen/BSS/recordings/01wo0316.47zc0303..wav"
# ref_data = wavfile.read(ref_file)[1]
# ref_data = np.array(ref_data).astype(np.float64).T

time_offset_sec = 0.3

if __name__ == "__main__":
    choices = ["fastmnmf","ilrma","fastmnmf2",]

    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstration of blind source separation using "
        "IVA, ILRMA, sparse IVA, FastMNMF, or FastMNMF2 ."
    )
    parser.add_argument("-b", "--block", type=int, default=2048, help="STFT block size")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        choices=choices,
        default=None,
        help="Chooses BSS method to run",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Saves the output of the separation to wav files",
    )
    parser.add_argument(
        "--SCM_index",
        type=int,
        default=0
    )
    args = parser.parse_args()

    import pickle
    W = None
    if args.SCM_index == 0:
        scm_file = open("scm.tmp", "wb")
    elif args.SCM_index>0:
        from load_scm import loadSCM
        for i in range(args.SCM_index):
            W = loadSCM()
            if W is None:
                print("SCM index out of range, using ilrma to initialize SCM")
                args.SCM_index==0
    print(args.SCM_index)
    import pyroomacoustics as pra

    ## START BSS
    def bss(wav, W=W):
        t_begin = time.perf_counter()
        wav = wav.numpy()
        bss_type = args.algo
        if args.SCM_index==0:
            print("Initialize with ILRMA")
            Y, W = pra.bss.ilrma(
                X, n_iter=50, n_components=4, proj_back=True, return_filters=True,
            )
            pickle.dump(W, scm_file)
        else:
            print("Use {}-th SCM".format(args.SCM_index))
            
        if bss_type == "ilrma":
            print("Run ILRMA")
            Y = pra.bss.ilrma(
                X, n_iter=100, n_components=32, proj_back=True, W0=W,
            )
        elif bss_type == "fastmnmf":
            print("Run FastMNMF")
            Y = pra.bss.fastmnmf(
                X, n_iter=100, n_components=32, n_src=4, W0=W,
            )
        elif bss_type == "fastmnmf2":
            print("Run FastMNMF2")
            Y = pra.bss.fastmnmf2(
                X, n_iter=100, n_components=32, n_src=4, W0=W,
            )

        t_end = time.perf_counter()
        print("Time for BSS: {:.2f} s".format(t_end - t_begin))

        ## STFT Synthesis
        y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
        # assert not np.isnan(Y).any(), "spec includes NaN"
        # y = MultiISTFT(Y, shape="MFT").to(torch.float32)
        # y = y.numpy()
        return y

    ## Prepare one-shot STFT
    L = args.block
    hop = L // 2
    win_a = pra.hann(L)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

    total_wer = []
    total_ins = []
    total_sub = []
    total_del = []
    total_all = []

    ground_truth_wer = []
    for wav_file in wav_files:
        print('Separating ', wav_file)
        fs, data = wavfile.read(dir+wav_file)
        data = data[:,1:5]
        data = np.array(data)

        ## STFT ANALYSIS
        X = pra.transform.stft.analysis(data, L, hop, win=win_a)
        wav = torch.Tensor(data)

        t_begin = time.perf_counter()

        ## Perform WPE
        from nara_wpe.torch_wpe import wpe_v6
        from nara_wpe.wpe import wpe
        from nara_wpe.utils import stft, istft

        stft_options = dict(size=512, shift=128)
        sampling_rate = 16000
        delay = 3
        iterations = 10
        taps = 10
        alpha=0.9999
        Y = stft(data.T, **stft_options).transpose(2, 0, 1)

        # PyTorch WPE
        Y_ = torch.from_numpy(Y).cuda()
        Z_ = wpe_v6(
            Y_,
            taps=taps,
            delay=delay,
            iterations=iterations,
            statistics_mode='full'
        )
        Z_ = Z_.cpu()

        # NumPy WPE
        # Z = wpe(
        #     Y,
        #     taps=taps,
        #     delay=delay,
        #     iterations=iterations,
        #     statistics_mode='full'
        # )

        wav_wpe = istft(Z_.permute(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])
        wav_wpe = torch.from_numpy(wav_wpe).transpose(0, 1)

        y = bss(wav_wpe)
        y = y.T
        
        if args.save:
            from scipy.io import wavfile
            from wer import wer
            from asr import asr
            hyps = []
            for i, sig in enumerate(y):
                # wavfile.write(
                #     "temp{}.wav".format(i),
                #     fs,
                #     pra.normalize(sig, bits=16).astype(np.int16),
                # )
                hyps.append(asr(sig))
            print(hyps)
            for f in wav_file.split('.')[:-1]:
                m = 0
                w = 300.0
                I = 0
                S = 0
                D = 0
                A = 0

                print('\nGround truth:')
                ground_truth_wer.append(wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=asr('/n/work1/shi/corpus/wsj/{}.wav'.format(f)), hyp_fid=f)[0])
                
                print('Matching {}'.format(f))
                for i, hyp in enumerate(hyps):
                    w_, ins_, sub_, del_, all_ = wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=hyp, hyp_fid=f)
                    if w_ != None and w_ < w:
                        w = w_
                        m = i
                        I = ins_
                        S = sub_
                        D = del_
                        A = all_
                total_wer.append(w)
                wavfile.write(
                    "{}/{}.wav".format(args.algo, f),
                    fs,
                    pra.normalize(y[m], bits=16).astype(np.int16),
                )

            print('Average WER: ', sum(total_wer) / len(total_wer))
            print('Weighted average WER: ', (sum(total_ins) + sum(total_sub) + sum(total_del)) / sum(total_all))
            print('Total ground truth WER: ', sum(ground_truth_wer) / len(ground_truth_wer))