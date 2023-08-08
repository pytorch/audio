import torch
import random
import torchaudio.functional as F


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_dataset,
        sampling_rate = 16000,
        snr = (10, 20),
        p = 0.5,
        seed: int = 42,
    ):
        # `noise_dataset` is a pytorch dataset that contains noise samples
        # `p` is the probability of adding any noise

        super().__init__()
        self.noise_dataset = noise_dataset
        self.noise_count = len(noise_dataset)
        self.sampling_rate = sampling_rate
        self.snr = snr
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)
        self.noise_batch = None
        self.position = 0


    def fetch_noise_batch(self, total_length):
        # This will fetch noise until the number of sample points is equal or more than the speech samples

        wav_list = []
        while total_length > 0:
            idx = self.rng.randint(0, self.noise_count - 1)  # (both included)
            wav, sr, filename = self.noise_dataset[idx]
            assert sr == self.sampling_rate
            wav_list.append(wav)
            total_length -= wav.size(-1)
        self.noise_batch = torch.cat(wav_list, dim=-1)
        self.position = self.rng.randint(0, self.noise_batch.size(-1) - 1)
        return self.noise_batch


    def get_my_noise(self, length):
        noise_list = []
        while length > 0:
            noise = self.noise_batch[:, self.position:self.position+length]
            noise_list.append(noise)
            length -= noise.size(-1)
            self.position += length
            if self.position > self.noise_batch.size(-1):
                self.position = 0
        return torch.cat(noise_list, dim=-1)


    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # You might need to resample the input or noise such that their 
        # sample rates are matched.
        # We don't need to do this here for Librispeech, because
        # Librispeech and Musan are both 16kHz.

        # Reference: https://github.com/lhotse-speech/lhotse/blob/master/lhotse/cut/set.py#L1798

        if sample.ndim == 1:
            sample = sample.unsqueeze(0)

        if self.rng.uniform(0.0, 1.0) > self.p:
            return sample[0], sample.size(-1)

        snr = self.rng.uniform(*self.snr) if isinstance(self.snr, (list, tuple)) else self.snr
        noise = self.get_my_noise(sample.size(-1))
        noisy_sample = F.add_noise(sample, noise, torch.tensor([snr]))
        return noisy_sample[0], noisy_sample.size(-1)