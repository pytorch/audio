import random
from typing import List

import sentencepiece as spm
import torch
import torchvision
from data_module import LRS3DataModule
from lightning import Batch
from lightning_av import AVBatch


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        cloned = x.clone()
        length = cloned.size(1)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[:, t_start:t_end] = 0
        return cloned


def _extract_labels(sp_model, samples: List):
    targets = [sp_model.encode(sample[-1].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


def _extract_features(video_pipeline, audio_pipeline, samples, args):
    raw_videos = []
    raw_audios = []
    for sample in samples:
        if args.modality == "visual":
            raw_videos.append(sample[0])
        if args.modality == "audio":
            raw_audios.append(sample[0])
        if args.modality == "audiovisual":
            length = min(len(sample[0]) // 640, len(sample[1]))
            raw_audios.append(sample[0][: length * 640])
            raw_videos.append(sample[1][:length])

    if args.modality == "visual" or args.modality == "audiovisual":
        videos = torch.nn.utils.rnn.pad_sequence(raw_videos, batch_first=True)
        videos = video_pipeline(videos)
        video_lengths = torch.tensor([elem.shape[0] for elem in videos], dtype=torch.int32)
    if args.modality == "audio" or args.modality == "audiovisual":
        audios = torch.nn.utils.rnn.pad_sequence(raw_audios, batch_first=True)
        audios = audio_pipeline(audios)
        audio_lengths = torch.tensor([elem.shape[0] // 640 for elem in audios], dtype=torch.int32)
    if args.modality == "visual":
        return videos, video_lengths
    if args.modality == "audio":
        return audios, audio_lengths
    if args.modality == "audiovisual":
        return audios, videos, audio_lengths, video_lengths


class TrainTransform:
    def __init__(self, sp_model_path: str, args):
        self.args = args
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.train_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.RandomCrop(88),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            torchvision.transforms.Grayscale(),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            AdaptiveTimeMask(10, 25),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
        self.train_audio_pipeline = torch.nn.Sequential(
            AdaptiveTimeMask(10, 25),
        )

    def __call__(self, samples: List):
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        if self.args.modality == "audio":
            audios, audio_lengths = _extract_features(
                self.train_video_pipeline, self.train_audio_pipeline, samples, self.args
            )
            return Batch(audios, audio_lengths, targets, target_lengths)
        if self.args.modality == "visual":
            videos, video_lengths = _extract_features(
                self.train_video_pipeline, self.train_audio_pipeline, samples, self.args
            )
            return Batch(videos, video_lengths, targets, target_lengths)
        if self.args.modality == "audiovisual":
            audios, videos, audio_lengths, video_lengths = _extract_features(
                self.train_video_pipeline, self.train_audio_pipeline, samples, self.args
            )
            return AVBatch(audios, videos, audio_lengths, video_lengths, targets, target_lengths)


class ValTransform:
    def __init__(self, sp_model_path: str, args):
        self.args = args
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.valid_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            torchvision.transforms.Grayscale(),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
        self.valid_audio_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x),
        )

    def __call__(self, samples: List):
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        if self.args.modality == "audio":
            audios, audio_lengths = _extract_features(
                self.valid_video_pipeline, self.valid_audio_pipeline, samples, self.args
            )
            return Batch(audios, audio_lengths, targets, target_lengths)
        if self.args.modality == "visual":
            videos, video_lengths = _extract_features(
                self.valid_video_pipeline, self.valid_audio_pipeline, samples, self.args
            )
            return Batch(videos, video_lengths, targets, target_lengths)
        if self.args.modality == "audiovisual":
            audios, videos, audio_lengths, video_lengths = _extract_features(
                self.valid_video_pipeline, self.valid_audio_pipeline, samples, self.args
            )
            return AVBatch(audios, videos, audio_lengths, video_lengths, targets, target_lengths)


class TestTransform:
    def __init__(self, sp_model_path: str, args):
        self.val_transforms = ValTransform(sp_model_path, args)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(args, sp_model_path, max_frames=1800):
    train_transform = TrainTransform(sp_model_path=sp_model_path, args=args)
    val_transform = ValTransform(sp_model_path=sp_model_path, args=args)
    test_transform = TestTransform(sp_model_path=sp_model_path, args=args)
    return LRS3DataModule(
        args=args,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        max_frames=max_frames,
    )
