import torch
from parameterized import parameterized
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    TempDirMixin,
    load_params,
    skipIfNoExec,
    get_sinusoid,
    get_sine_sweep,
    save_wav,
)
from torchaudio_unittest.common_utils.kaldi_utils import (
    convert_args,
    run_kaldi,
)


def _get_waveform(sample_rate, frame_shift, frame_length):
    # Note:
    # If `subtract_mean` is enabled, Kaldi features subtract mean value computed
    # over time axis. This accumulates the frame-wise errors. Thus, makding duration
    # longer makes the error Tensor mismatch bigger.
    # For the sake of compatibility testing, just a few hops of window should be sufficient.
    window_shift = int(sample_rate * 0.001 * frame_shift)
    window_size = int(sample_rate * 0.001 * frame_length)

    num_frames = window_size + 6 * window_shift
    # duration = num_frames / sample_rate
    # waveform = get_sinusoid(sample_rate=sample_rate, n_channels=2, duration=duration, frequency=600)
    waveform = get_sine_sweep(sample_rate=sample_rate, num_frames=num_frames)
    print(waveform.min(), waveform.max())
    # print(waveform.shape)
    # waveform = get_whitenoise(sample_rate=sample_rate, n_channels=2, duration=duration)
    return waveform


def _get_frame_opts(sample_rate, kwargs):
    from torchaudio._kaldi.feature_window import FrameExtractionOptions

    return FrameExtractionOptions(
        samp_freq=sample_rate,
        blackman_coeff=kwargs["blackman_coeff"],
        dither=None if kwargs["dither"] == 0 else kwargs["dither"],
        frame_shift_ms=kwargs["frame_shift"],
        frame_length_ms=kwargs["frame_length"],
        preemph_coeff=kwargs["preemphasis_coefficient"],
        remove_dc_offset=kwargs["remove_dc_offset"],
        round_to_power_of_two=kwargs["round_to_power_of_two"],
        snip_edges=kwargs["snip_edges"],
        window_type=kwargs["window_type"],
    )


def _get_mel_opts(kwargs):
    from torchaudio._kaldi.mel_computations import MelBanksOptions

    return MelBanksOptions(
        num_bins=kwargs["num_mel_bins"],
        low_freq=kwargs["low_freq"],
        high_freq=kwargs["high_freq"],
        vtln_low=kwargs["vtln_low"],
        vtln_high=kwargs["vtln_high"],
    )


class KaldiRef(TempDirMixin, TestBaseMixin):
    def _plot(self, extractor, waveform, sample_rate, result, kaldi_result):
        import matplotlib.pyplot as plt

        num_channel = waveform.shape[0]
        f, ax = plt.subplots(5, max(num_channel, 2))  # , sharex=True, sharey=True)
        for c in range(num_channel):
            ax[0][c].set_title("input")
            ax[0][c].plot(waveform[c])
            shift = extractor._computer.frame_options.window_shift
            size = extractor._computer.frame_options.window_size
            noverlap = size - shift
            ax[1][c].specgram(waveform[c], Fs=sample_rate, NFFT=size, noverlap=noverlap)
            ax[2][c].set_title("torchaudio")
            ax[2][c].imshow(result[c].T, aspect="auto", interpolation="nearest")
            ax[3][c].set_title("kaldi")
            ax[3][c].imshow(kaldi_result[c].T, aspect="auto", interpolation="nearest")
            ax[4][c].set_title("Diff")
            diff = (result[c].T - kaldi_result[c].T).abs()
            diff = ax[4][c].imshow(diff, aspect="auto", interpolation="nearest")
            plt.colorbar(diff, ax=ax[4][c])
        plt.savefig(f"build/{self.id()}.png")

    def _test(self, extractor, command, waveform, sample_rate, atol=1e-5, rtol=1e-3):
        assert waveform.ndim == 2, "Expected waveform to have shape [channel, time]"

        # Format the wave into proper int16 once.
        waveform = waveform * (2 ** 15 - 1)
        print(waveform.min(), waveform.max())
        waveform = waveform.round()
        print(waveform.min(), waveform.max())
        waveform = waveform.clamp(min=-32768, max=32767)
        print(waveform.min(), waveform.max())
        waveform = waveform.to(torch.int16)
        print(waveform.min(), waveform.max())
        try:
            # TODO: get rid of the  for torchaudio implementation
            result = extractor.compute(waveform.to(torch.float32))
            print("result:", result)
        except Exception as e:
            print(e)

        kaldi_results = []
        for i, wv in enumerate(waveform.unsqueeze(1)):
            path = self.get_temp_path(f"channel_{i}.wav")
            save_wav(path, wv, sample_rate)
            try:
                kaldi_results.append(run_kaldi(command, "scp", path))
            except Exception as e:
                print(e)
        kaldi_result = torch.stack(kaldi_results)
        print("kaldi_result:", kaldi_result)

        try:
            self.assertEqual(result, kaldi_result, atol=atol, rtol=rtol)
        except Exception:
            self._plot(extractor, waveform, sample_rate, result, kaldi_result)
            raise

    # TODO: revise the parameters.
    # TODO: add cases where some parameters are ineffective, such as energy_floor
    @parameterized.expand(load_params("kaldi_test_spectrogram_args.jsonl"))
    @skipIfNoExec("compute-spectrogram-feats")
    def test_spectrogram_ref(self, kwargs):
        from torchaudio._kaldi.feature_common import OfflineFeatureTpl
        from torchaudio._kaldi.feature_spectrogram import SpectrogramComputer, SpectrogramOptions

        sample_rate = 16000
        opts = SpectrogramOptions(
            frame_opts=_get_frame_opts(sample_rate, kwargs),
            energy_floor=None if kwargs["energy_floor"] == 0 else kwargs["energy_floor"],
            raw_energy=kwargs["raw_energy"],
            return_raw_fft=kwargs.get("return_raw_fft", False),  # TODO: add True
        )
        extractor = OfflineFeatureTpl(SpectrogramComputer(opts), subtract_mean=kwargs["subtract_mean"])
        command = ["compute-spectrogram-feats"] + convert_args(**kwargs) + ["scp:-", "ark:-"]

        waveform = _get_waveform(
            sample_rate=sample_rate,
            frame_shift=kwargs["frame_shift"],
            frame_length=kwargs["frame_length"],
        )

        # When subtract_mean is enabled, it computes the mean across time, which
        # aggregates the errors on each time frame, and it makes it very difficult to
        # get better numerical parity.
        atol = 5e-5 if kwargs["subtract_mean"] else 1e-5

        self._test(extractor, command, waveform, sample_rate, atol=atol)

    # TODO: revise the parameters.
    # TODO: add cases where some parameters are ineffective, such as energy_floor, vtln_warp
    @parameterized.expand(load_params("kaldi_test_fbank_args.jsonl"))
    @skipIfNoExec("compute-fbank-feats")
    def test_fbank_ref(self, kwargs):
        from torchaudio._kaldi.feature_common import OfflineFeatureTpl
        from torchaudio._kaldi.feature_fbank import FbankComputer, FbankOptions

        # temp: overwrite while we resolve other issues
        kwargs["use_log_fbank"] = True
        # htk_compat is not supported yet
        kwargs["htk_compat"] = False

        sample_rate = 16000
        opts = FbankOptions(
            frame_opts=_get_frame_opts(sample_rate, kwargs),
            mel_opts=_get_mel_opts(kwargs),
            use_energy=kwargs["use_energy"],
            energy_floor=kwargs["energy_floor"],
            raw_energy=kwargs["raw_energy"],
            use_log_fbank=kwargs["use_log_fbank"],
            use_power=kwargs["use_power"],
        )
        extractor = OfflineFeatureTpl(
            FbankComputer(opts, vtln_warp=None if kwargs["vtln_warp"] == 0 else kwargs["vtln_warp"]),
            subtract_mean=kwargs["subtract_mean"],
        )
        command = ["compute-fbank-feats"] + convert_args(**kwargs) + ["scp:-", "ark:-"]

        waveform = _get_waveform(
            sample_rate=sample_rate,
            frame_shift=kwargs["frame_shift"],
            frame_length=kwargs["frame_length"],
        )

        # When `subtract_mean` is enabled, it computes the mean across time, which
        # aggregates the errors on each time frame, and it makes it very difficult to
        # get better numerical parity.
        atol = 5e-5 if kwargs["subtract_mean"] else 1e-5

        self._test(extractor, command, waveform, sample_rate, atol=atol)

    # TODO: revise the parameters.
    # TODO: add cases where some parameters are ineffective, such as energy_floor, vtln_warp
    @parameterized.expand(load_params("kaldi_test_mfcc_args.jsonl"))
    @skipIfNoExec("compute-mfcc-feats")
    def test_mfcc_ref(self, kwargs):
        from torchaudio._kaldi.feature_common import OfflineFeatureTpl
        from torchaudio._kaldi.feature_mfcc import MfccComputer, MfccOptions

        # htk_compat is not supported yet
        kwargs["htk_compat"] = False

        sample_rate = 16000
        opts = MfccOptions(
            frame_opts=_get_frame_opts(sample_rate, kwargs),
            mel_opts=_get_mel_opts(kwargs),
            num_ceps=kwargs["num_ceps"],
            use_energy=kwargs["use_energy"],
            energy_floor=kwargs["energy_floor"],
            raw_energy=kwargs["raw_energy"],
            cepstral_lifter=kwargs["cepstral_lifter"],
        )
        extractor = OfflineFeatureTpl(
            MfccComputer(opts, vtln_warp=None if kwargs["vtln_warp"] == 0 else kwargs["vtln_warp"]),
            subtract_mean=kwargs["subtract_mean"],
        )
        command = ["compute-mfcc-feats"] + convert_args(**kwargs) + ["scp:-", "ark:-"]

        waveform = _get_waveform(
            sample_rate=sample_rate,
            frame_shift=kwargs["frame_shift"],
            frame_length=kwargs["frame_length"],
        )

        self._test(extractor, command, waveform, sample_rate, atol=5e-5)
