import os, random
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
from FastMNMF1 import FastMNMF1
from Base import MultiSTFT, MultiISTFT
import torch
import torchaudio
import numpy as np
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile
from wer import wer
from asr import asr

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

dir = "/n/work1/juchen/BSS/recordings_noisy_noisy/"
# dir = "/n/work1/shi/corpus/wham/whamr_2ch/wav8k/max/cv/mix_both_reverb/"
all_wav_files = os.listdir(dir)
wav_files = random.sample(all_wav_files, 5)
# wav_files = all_wav_files
wav_files[0] = "46rc0208.46wc030o.wav"
# wav_files[0] = "47nc0404.473c040t.wav"

# ref_file = "/n/work1/juchen/BSS/recordings/01wo0316.47zc0303..wav"
# ref_data = wavfile.read(ref_file)[1]
# ref_data = np.array(ref_data).astype(np.float64).T

# time_offset_sec = 0.3

fs = 16000

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
        default=choices[0],
        help="Chooses BSS method to run",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Saves the output of the separation to wav files",
    )
    parser.add_argument(
        "--n_init_wav",
        type=int,
        default=20,
    )
    # parser.add_argument(
    #     "--SCM_index",
    #     type=int,
    #     default=0
    # )
    args = parser.parse_args()

    # import pickle
    # W = None
    # if args.SCM_index>0:
    #     file = open("scm_noisy-13-15.tmp", "rb")
    #     def loadSCM():
    #         try:
    #             scm = pickle.load(file)
    #             return scm
    #         except:
    #             return None
    #     for i in range(args.SCM_index):
    #         W = loadSCM()
    #     if W is None:
    #         print("SCM index out of range, using ilrma to initialize SCM")
    #         args.SCM_index=0
    # elif args.SCM_index<0:
    #     print("No initialization")
    # if args.SCM_index == 0:
    #     scm_file = open("scm.tmp", "wb")
    # print(args.SCM_index)

    import pyroomacoustics as pra

    def wpe(data):
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
        return wav_wpe

    def bss_init(wav, fs=fs, n_source=3, n_iter=100, device='cuda', block=2048, method="fastmnmf"):
        spec_FTM = MultiSTFT(wav[:, :], n_fft=block)

        separater = FastMNMF1(
            n_source=n_source,
            n_basis=2,
            device=device,
            init_SCM="circular",
            n_bit=64,
            g_eps=5e-2,
        )
        separater.load_spectrogram(spec_FTM, fs)
        separater.solve(n_iter=n_iter, save_wav=False)

        return separater.Q_FMM, separater.G_NFM

    ## START BSS
    def bss(wav, fs=fs, n_source=3, n_iter=50, device='cuda', block=2048, Q_FMM=None, G_NFM=None):
        ## Prepare one-shot STFT
        # L = args.block
        # hop = L // 2
        # win_a = pra.hann(L)
        # win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)
        t_begin = time.perf_counter()
        # wav = wav.numpy()
        # bss_type = args.algo
        # X = pra.transform.stft.analysis(wav, L, hop, win=win_a)
        spec_FTM = MultiSTFT(wav[:, :], n_fft=block)
        print(type(spec_FTM))

        if Q_FMM==None or G_NFM==None:
            separater = FastMNMF1(
                n_source=n_source,
                n_basis=32,
                device=device,
                init_SCM="twostep",
                n_bit=64,
                algo="IP",
                n_iter_init=50,
                g_eps=5e-2,
            )

            separater.file_id = None
            separater.load_spectrogram(spec_FTM, fs)
            separater.solve(
                n_iter=n_iter,
                save_dir="./",
                save_likelihood=False,
                save_param=False,
                save_wav=False,
                interval_save=5,
            )
        else:
            separater = FastMNMF1(
                n_source=n_source,
                n_basis=32,
                device=device,
                init_SCM="manually_twostep",
                n_bit=64,
                algo="IP",
                g_eps=5e-2,
            )
            separater.Q_FMM = Q_FMM
            separater.G_NFM = G_NFM
            separater.file_id = None
            separater.load_spectrogram(spec_FTM, fs)
            separater.solve(
                n_iter=n_iter,
                save_dir="./",
                save_likelihood=False,
                save_param=False,
                save_wav=False,
                interval_save=5,
            )

            # separater.file_id = None
            # separater.load_spectrogram(spec_FTM, fs)
            # separater.solve(
            #     n_iter=100,
            #     save_dir="./",
            #     save_likelihood=False,
            #     save_param=False,
            #     save_wav=False,
            #     interval_save=5,
            # )
            
        Y_mnmf = separater.separated_spec.cpu().numpy()

        t_end = time.perf_counter()
        print("Time for BSS: {:.2f} s".format(t_end - t_begin))

        wtsum_steering_vector, eigen_steering_vector, scm = separater.wtsum_steering_vector, separater.eigen_steering_vector, separater.scm
        Y_wtsum_mvdr = mvdrbf(spec_FTM, wtsum_steering_vector, scm)
        Y_eigen_mvdr = mvdrbf(spec_FTM, eigen_steering_vector, scm)
        
        # print(Y.shape)
        # print('Y_bf shape: ', Y_bf.shape)

        # Permutation solver
        from pb_bss.permutation_alignment import apply_mapping, DHTVPermutationAlignment as DHTV
        ps = DHTV(stft_size=block, segment_start=0, segment_width=512, segment_shift=64, main_iterations=20, sub_iterations=2, similarity_metric='cos')
        mapping = ps.calculate_mapping(mask = abs(Y_mnmf))
        Y_mnmf = apply_mapping(Y_mnmf, mapping)

        # print('Y shape', Y.shape)
        Y_mnmf = torch.from_numpy(Y_mnmf)
        Y_wtsum_mvdr = torch.from_numpy(Y_wtsum_mvdr)
        Y_eigen_mvdr = torch.from_numpy(Y_eigen_mvdr)


        t_end = time.perf_counter()
        print("Time for BSS: {:.2f} s".format(t_end - t_begin))
        print(f'Computational time / Duration: {(t_end - t_begin) / (wav.shape[0] / fs)}')

        ## STFT Synthesis
        # y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
        assert not np.isnan(Y_mnmf).any(), "spec includes NaN"
        y_mnmf = MultiISTFT(Y_mnmf, shape="MFT").numpy()
        y_wtsum_mvdr = MultiISTFT(Y_wtsum_mvdr, shape="MFT").numpy()
        y_eigen_mvdr = MultiISTFT(Y_eigen_mvdr, shape="MFT").numpy()

        return y_mnmf, y_wtsum_mvdr, y_eigen_mvdr

    def mvdrbf(specs, steering_vector, scm):
        # Input: sepcs: F*T*M, steering_vector = N*M*F, scm = N*F*M*M
        n_source = scm.shape[0]
        print('scm: ', scm.shape)
        print('steering vector: ', steering_vector.shape)
        def get_mvdr_beamformer(steering_vector, R):
            number_of_mic = steering_vector.shape[0]
            n_frequency_grid = steering_vector.shape[1]
            # frequency_grid = np.linspace(0, steering_vector.shape[1], len(steering_vector.shape[1]))
            # frequency_grid = frequency_grid[0:int(self.fft_length / 2) + 1]        
            beamformer = np.ones((number_of_mic, n_frequency_grid), dtype=np.complex64)
            for f in range(0, n_frequency_grid):
                # R_cut = np.reshape(R[:, :, f], [number_of_mic, number_of_mic])
                R_cut = R[f]
                inv_R = np.linalg.pinv(R_cut)
                a = np.matmul(np.conjugate(steering_vector[:, f]), inv_R)
                b = np.matmul(a, steering_vector[:, f])
                b = np.reshape(b, [1, 1])
                beamformer[:, f] = np.matmul(inv_R, steering_vector[:, f]) / b # number_of_mic *1   = number_of_mic *1 vector/scalar        
            return beamformer
        
        def apply_beamformer(beamformer, complex_spectrum):
            number_of_bins, number_of_frames, number_of_channels = np.shape(complex_spectrum)        
            enhanced_spectrum = np.zeros(( number_of_bins, number_of_frames), dtype=np.complex64)
            for f in range(0, number_of_bins):
                enhanced_spectrum[f, :] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[f].T)
            # return util.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)        
            return enhanced_spectrum
        
        separated_spec = []
        for i in range(n_source):
            beamformer = get_mvdr_beamformer(
                steering_vector=steering_vector[i].T.cpu(),
                R=np.sum(scm.cpu().numpy(), axis=0)
                #   - scm[i].cpu().numpy()
            )
            # print('## ', (np.sum(scm.cpu().numpy(), axis=0)-scm[i].cpu().numpy()).shape)
            separated_spec.append(apply_beamformer(beamformer=beamformer, complex_spectrum=specs))
        separated_spec = np.array(separated_spec)
        print(separated_spec.shape)
        return separated_spec


    hyp_mixture_wer = []
    hyp_mixture_ins = []
    hyp_mixture_sub = []
    hyp_mixture_del = []
    hyp_mixture_all = []

    hyp_fastmnmf_wer = []
    hyp_fastmnmf_ins = []
    hyp_fastmnmf_sub = []
    hyp_fastmnmf_del = []
    hyp_fastmnmf_all = []

    hyp_wtsum_mvdr_wer = []
    hyp_wtsum_mvdr_ins = []
    hyp_wtsum_mvdr_sub = []
    hyp_wtsum_mvdr_del = []
    hyp_wtsum_mvdr_all = []

    hyp_eigen_mvdr_wer = []
    hyp_eigen_mvdr_ins = []
    hyp_eigen_mvdr_sub = []
    hyp_eigen_mvdr_del = []
    hyp_eigen_mvdr_all = []

    gt_wer = []
    gt_ins = []
    gt_sub = []
    gt_del = []
    gt_all = []

    cnt = 0


    init_wav_files = random.sample(all_wav_files, args.n_init_wav)
    init_wav = np.concatenate(
        [
            wavfile.read(dir+wav_file)[1][:,1:5] for wav_file in init_wav_files
        ],
        axis = 0
        )
    print("init_wav shape:", init_wav.shape)
    wpe_init_wav = wpe(init_wav)
    print("wpe_init_wav shape:", wpe_init_wav.shape)
    Q, G = bss_init(wpe_init_wav)

    for idx, wav_file in enumerate(wav_files):
        cnt+=1
        print('{}: Separating '.format(cnt), wav_file)
        data = wavfile.read(dir+wav_file)[1]
        print(data.shape)
        data = data[:,1:5]
        data = np.array(data)
        # data = torch.Tensor(data)

        # Plot mixture
        plt.figure()
        plt.subplot(2,2,1)
        plt.specgram(data[:, 0], Fs=fs, scale_by_freq=True, sides = 'default')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2,2,2)
        plt.specgram(data[:, 1], Fs=fs, scale_by_freq=True, sides = 'default')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2,2,3)
        plt.specgram(data[:, 2], Fs=fs, scale_by_freq=True, sides = 'default')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2,2,4)
        plt.specgram(data[:, 3], Fs=fs, scale_by_freq=True, sides = 'default')
        plt.colorbar(format='%+2.0f dB')

        plt.savefig("mixture.png")
        plt.close()

        t_begin = time.perf_counter()
        
        # Separation
        wav_wpe = wpe(data)
        y_mnmf, y_wtsum_mvdr, y_eigen_mvdr = bss(wav_wpe, Q_FMM = Q, G_NFM=G)
        # print('Wav shape: ', y.shape)
        
        if args.save:
            from scipy.io import wavfile
            from wer import wer
            from asr import asr
            hyps_mixture = []
            hyps_fastmnmf = []
            hyps_wtsum_mvdr = []
            hyps_eigen_mvdr = []

            # Hypothesis transcription of separated speech
            for i, sig in enumerate(y_mnmf):
                # wavfile.write(
                #     "temp{}.wav".format(i),
                #     fs,
                #     pra.normalize(sig, bits=16).astype(np.int16),
                # )
                hyps_mixture.append(asr(data[:, 0]))
                hyps_fastmnmf.append(asr(sig))
            for i, sig in enumerate(y_wtsum_mvdr):
                # wavfile.write(
                #     "temp{}.wav".format(i),
                #     fs,
                #     pra.normalize(sig, bits=16).astype(np.int16),
                # )
                hyps_wtsum_mvdr.append(asr(sig))
            for i, sig in enumerate(y_eigen_mvdr):
                # wavfile.write(
                #     "temp{}.wav".format(i),
                #     fs,
                #     pra.normalize(sig, bits=16).astype(np.int16),
                # )
                hyps_eigen_mvdr.append(asr(sig))
            for f in wav_file.split('.')[:-1]:
                print('\nGround truth:')
                gt_w_, gt_ins_, gt_sub_, gt_del_, gt_all_ = wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=asr('/n/work1/shi/corpus/wsj/{}.wav'.format(f)), hyp_fid=f)
                gt_wer.append(gt_w_)
                gt_ins.append(gt_ins_)
                gt_sub.append(gt_sub_)
                gt_del.append(gt_del_)
                gt_all.append(gt_all_)

                # WERs on the original mixture
                print('Matching {}'.format(f))
                print('Mixture:')
                
                m = 0
                w = 300.0
                I = 0
                S = 0
                D = 0
                A = 0
                for i, hyp in enumerate(hyps_mixture):
                    w_, ins_, sub_, del_, all_ = wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=hyp, hyp_fid=f)
                    if w_ != None and w_ < w:
                        w = w_
                        m = i
                        I = ins_
                        S = sub_
                        D = del_
                        A = all_

                hyp_mixture_wer.append(w)
                hyp_mixture_ins.append(I)
                hyp_mixture_sub.append(S)
                hyp_mixture_del.append(D)
                hyp_mixture_all.append(A)

                # WERs with FastMNMF only
                print('Matching {}'.format(f))
                print('FastMNMF:')
                
                m = 0
                w = 300.0
                I = 0
                S = 0
                D = 0
                A = 0
                for i, hyp in enumerate(hyps_fastmnmf):
                    w_, ins_, sub_, del_, all_ = wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=hyp, hyp_fid=f)
                    if w_ != None and w_ < w:
                        w = w_
                        m = i
                        I = ins_
                        S = sub_
                        D = del_
                        A = all_

                hyp_fastmnmf_wer.append(w)
                hyp_fastmnmf_ins.append(I)
                hyp_fastmnmf_sub.append(S)
                hyp_fastmnmf_del.append(D)
                hyp_fastmnmf_all.append(A)

                wavfile.write(
                    f'bss_output/{idx}_{f}_fastmnmf.wav',
                    fs,
                    pra.normalize(y_mnmf[m], bits=16).astype(np.int16),
                )

                # WERs after MVDR using weighted sum of pesudo-steering-vectors
                print('MVDR(wtsum)')

                m = 0
                w = 300.0
                I = 0
                S = 0
                D = 0
                A = 0

                for i, hyp in enumerate(hyps_wtsum_mvdr):
                    w_, ins_, sub_, del_, all_ = wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=hyp, hyp_fid=f)
                    if w_ != None and w_ < w:
                        w = w_
                        m = i
                        I = ins_
                        S = sub_
                        D = del_
                        A = all_

                hyp_wtsum_mvdr_wer.append(w)
                hyp_wtsum_mvdr_ins.append(I)
                hyp_wtsum_mvdr_sub.append(S)
                hyp_wtsum_mvdr_del.append(D)
                hyp_wtsum_mvdr_all.append(A)

                wavfile.write(
                    f'bss_output/{idx}_{f}_fastmnmf+mvdr(wtsum).wav',
                    fs,
                    pra.normalize(y_wtsum_mvdr[m], bits=16).astype(np.int16),
                )

                # WERs after MVDR using principal eigenvector of G
                print('MVDR(eigen)')

                m = 0
                w = 300.0
                I = 0
                S = 0
                D = 0
                A = 0

                for i, hyp in enumerate(hyps_eigen_mvdr):
                    w_, ins_, sub_, del_, all_ = wer(char=1, v=1, ref_file="/n/work1/juchen/whisper_decoding/text", hyp=hyp, hyp_fid=f)
                    if w_ != None and w_ < w:
                        w = w_
                        m = i
                        I = ins_
                        S = sub_
                        D = del_
                        A = all_

                hyp_eigen_mvdr_wer.append(w)
                hyp_eigen_mvdr_ins.append(I)
                hyp_eigen_mvdr_sub.append(S)
                hyp_eigen_mvdr_del.append(D)
                hyp_eigen_mvdr_all.append(A)

                wavfile.write(
                    f'bss_output/{idx}_{f}_fastmnmf+mvdr(eigen).wav',
                    fs,
                    pra.normalize(y_eigen_mvdr[m], bits=16).astype(np.int16),
                )

                # plt.figure(figsize=(20, 5))
                # plt.subplot(1, 2, 1)
                # plt.specgram(wavfile.read('/n/work1/shi/corpus/wsj/{}.wav'.format(f))[1], Fs=fs, scale_by_freq=True, sides = 'default')
                # plt.title('Ground truth of {}'.format(f))
                # plt.subplot(1, 2, 2)
                # plt.specgram(y[m], Fs=fs, scale_by_freq=True, sides = 'default')
                # plt.title('Estimated {}'.format(f))
                # plt.savefig('{}_noisy_noisy/{}.png'.format(args.algo, f))
                # plt.close()

            # print('Average WER: ', sum(hyp_wer) / len(hyp_wer))
            print('Hypothesis WER on mixture: ', (sum(hyp_mixture_ins) + sum(hyp_mixture_sub) + sum(hyp_mixture_del)) * 100.0 / sum(hyp_mixture_all))
            print('Hypothesis WER by FastMNMF: ', (sum(hyp_fastmnmf_ins) + sum(hyp_fastmnmf_sub) + sum(hyp_fastmnmf_del)) * 100.0 / sum(hyp_fastmnmf_all))
            print('Hypothesis WER by FastMNMF+MVDR(wtsum): ', (sum(hyp_wtsum_mvdr_ins) + sum(hyp_wtsum_mvdr_sub) + sum(hyp_wtsum_mvdr_del)) * 100.0 / sum(hyp_wtsum_mvdr_all))
            print('Hypothesis WER by FastMNMF+MVDR(eigen): ', (sum(hyp_eigen_mvdr_ins) + sum(hyp_eigen_mvdr_sub) + sum(hyp_eigen_mvdr_del)) * 100.0 / sum(hyp_eigen_mvdr_all))
            print('Ground truth WER: ', (sum(gt_ins) + sum(gt_sub) + sum(gt_del)) * 100.0 / sum(gt_all))
    N = len(hyp_fastmnmf_wer)
    x = np.arange(N)
    width = 0.2
    plt.figure(figsize=(20, 10))
    plt.bar(x - 1.5 * width, gt_wer, width, label="Ground truth WER")
    plt.bar(x - 0.5 * width, hyp_fastmnmf_wer, width, label="Hypothesis WER by FastMNMF")
    plt.bar(x + 0.5 * width, hyp_wtsum_mvdr_wer, width, label="Hypothesis WER by FastMNMF+MVDR(wtsum)")
    # plt.bar(x + 1.5 * width, hyp_eigen_mvdr_wer, width, label="Hypothesis WER by FastMNMF+MVDR(eigen)")
    # plt.bar(x + 2 * width, hyp_mixture_wer, width, label="Hypothesis WER on mixture")
    plt.legend()
    plt.ylabel('WER')
    plt.xlabel('Index')
    plt.xticks(x)
    plt.ylim((0, 100))
    plt.title(f'Hypothesis WER by FastMNMF={(sum(hyp_fastmnmf_ins) + sum(hyp_fastmnmf_sub) + sum(hyp_fastmnmf_del)) * 100.0 / sum(hyp_fastmnmf_all)}\nHypothesis WER by FastMNMF+MVDR(wtsum)={(sum(hyp_wtsum_mvdr_ins) + sum(hyp_wtsum_mvdr_sub) + sum(hyp_wtsum_mvdr_del)) * 100.0 / sum(hyp_wtsum_mvdr_all)}\nHypothesis WER by FastMNMF+MVDR(eigen)={(sum(hyp_eigen_mvdr_ins) + sum(hyp_eigen_mvdr_sub) + sum(hyp_eigen_mvdr_del)) * 100.0 / sum(hyp_eigen_mvdr_all)}\nGround truth WER={(sum(gt_ins) + sum(gt_sub) + sum(gt_del)) * 100.0 / sum(gt_all)}\nHypothesis WER on mixtures={(sum(hyp_mixture_ins) + sum(hyp_mixture_sub) + sum(hyp_mixture_del)) * 100.0 / sum(hyp_mixture_all)}')
    plt.savefig(f'{args.algo}_WERs.png')