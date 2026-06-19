        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)
        rate = 2.0 ** (-float(n_steps) / bins_per_octave)
        self.orig_freq = int(sample_rate / rate)
        self.gcd = math.gcd(int(self.orig_freq), int(sample_rate))
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)
        self.pad = pad

    def forward(self, waveform: Tensor) -> Tensor:

        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.register_buffer("phase_advance", torch.linspace(0, math.pi * hop_length, n_freq)[..., None], persistent=False)

    def forward(self, complex_specgrams: Tensor, overriding_rate: Optional[float] = None) -> Tensor:
        r"""
                beta,
                dtype=dtype,
            )
            self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
            n_filter=self.n_filter,
            sample_rate=self.sample_rate,
        )
        self.register_buffer("filter_mat", filter_mat, persistent=False)

        dct_mat = F.create_dct(self.n_lfcc, self.n_filter, self.norm)
        self.register_buffer("dct_mat", dct_mat, persistent=False)
        self.log_lf = log_lf

    def forward(self, waveform: Tensor) -> Tensor:
        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError("Cannot select more MFCC coefficients than # mel bins")
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.register_buffer("dct_mat", dct_mat, persistent=False)
        self.log_mels = log_mels

    def forward(self, waveform: Tensor) -> Tensor:
        fb = F.melscale_fbanks(
            self.n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, melspec: Tensor) -> Tensor:
        r"""
        fb = F.melscale_fbanks(
            self.n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm, self.mel_scale
        )
        self.register_buffer("fb", fb, persistent=False)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)
        self.length = length
        self.power = power
        self.momentum = momentum
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)
        self.pad = pad
        self.normalized = normalized
        self.center = center
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer("window", window, persistent=False)
        self.pad = pad
        self.power = power
        self.normalized = normalized
