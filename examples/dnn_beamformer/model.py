import ci_sdr
import lightning.pytorch as pl
import torch
from asteroid.losses.stoi import NegSTOILoss
from asteroid.masknn import TDConvNet
from torchaudio.transforms import InverseSpectrogram, PSD, SoudenMVDR, Spectrogram


class DNNBeamformer(torch.nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, ref_channel: int = 0):
        super().__init__()
        self.stft = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mask_net = TDConvNet(
            n_fft // 2 + 1,
            2,
            out_chan=n_fft // 2 + 1,
            causal=False,
            mask_act="linear",
            norm_type="gLN",
        )
        self.beamformer = SoudenMVDR()
        self.psd = PSD()
        self.ref_channel = ref_channel

    def forward(self, mixture) -> torch.Tensor:
        spectrum = self.stft(mixture)  # (batch, channel, time, freq)
        batch, _, freq, time = spectrum.shape
        input_feature = torch.log(spectrum[:, self.ref_channel].abs() + 1e-8)  # (batch, freq, time)
        mask = torch.nn.functional.relu(self.mask_net(input_feature))  # (batch, 2, freq, time)
        mask_speech = mask[:, 0]
        mask_noise = mask[:, 1]
        psd_speech = self.psd(spectrum, mask_speech)
        psd_noise = self.psd(spectrum, mask_noise)
        enhanced_stft = self.beamformer(spectrum, psd_speech, psd_noise, self.ref_channel)
        enhanced_waveform = self.istft(enhanced_stft, length=mixture.shape[-1])
        return enhanced_waveform


class DNNBeamformerLightningModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super(DNNBeamformerLightningModule, self).__init__()
        self.model = model
        self.loss_stoi = NegSTOILoss(16000)

    def training_step(self, batch, batch_idx):
        mixture, clean = batch
        estimate = self.model(mixture)
        loss_cisdr = ci_sdr.pt.ci_sdr_loss(estimate, clean, compute_permutation=False, filter_length=512).mean()
        loss_stoi = self.loss_stoi(estimate, clean).mean()
        loss = loss_cisdr + loss_stoi * 10
        self.log("train/loss_cisdr", loss_cisdr.item())
        self.log("train/loss_stoi", loss_stoi.item())
        self.log("train/loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        mixture, clean = batch
        estimate = self.model(mixture)
        loss_cisdr = ci_sdr.pt.ci_sdr_loss(estimate, clean, compute_permutation=False, filter_length=512).mean()
        loss_stoi = self.loss_stoi(estimate, clean).mean()
        loss = loss_cisdr + loss_stoi * 10
        self.log("val/loss_cisdr", loss_cisdr.item())
        self.log("val/loss_stoi", loss_stoi.item())
        self.log("val/loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-8)
        return {
            "optimizer": optimizer,
        }
