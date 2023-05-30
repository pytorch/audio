#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torchvision
import numpy as np
from ibug.face_detection import RetinaFacePredictor
warnings.filterwarnings("ignore")


class LandmarksDetector:
    def __init__(self, device="cuda:0", model_name='resnet50'):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name)
        )

    def __call__(self, filename):
        video_frames = torchvision.io.read_video(filename, pts_unit='sec')[0].numpy()
        landmarks = []
        for frame in video_frames:
            detected_faces = self.face_detector(frame, rgb=False)
            if len(detected_faces) >= 1:
                landmarks.append(np.reshape(detected_faces[0][:4], (2, 2)))
            else:
                landmarks.append(None)
        return landmarks
