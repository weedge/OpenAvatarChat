

import os
import cv2
import numpy as np
import torch
from src.engine_utils.directory_info import DirectoryInfo
from src.handlers.avatar.liteavatar.algo.base_algo_adapter import BaseAlgoAdapter
from src.handlers.avatar.liteavatar.model.algo_model import (
    AvatarInitOption, AudioSlice, AvatarAlgoConfig,
    AvatarStatus, SignalType)


class SampleAdapter(BaseAlgoAdapter):

    def init(self, init_option: AvatarInitOption):
        data_dir = os.path.join(DirectoryInfo.get_project_dir(), "resource", "avatar", "preload")
        self._init_option = init_option
        self.bg_data_list = []
        bg_video = cv2.VideoCapture(f'{data_dir}/bg_video.mp4')
        while True:
            ret, img = bg_video.read()
            self.bg_data_list.append(img)
            if ret is False:
                break
        print(f"{len(self.bg_data_list)=}")

        bg_frame_cnt = 150
        self.bg_video_frame_count = len(
            self.bg_data_list) if bg_frame_cnt is None else min(
            bg_frame_cnt, len(
                self.bg_data_list))

        y1, y2, x1, x2 = open(f'{data_dir}/face_box.txt', 'r').readlines()[0].split()
        self.y1, self.y2, self.x1, self.x2 = int(y1), int(y2), int(x1), int(x2)

        self.merge_mask = (
            np.ones(
                (self.y2 -
                 self.y1,
                 self.x2 -
                 self.x1,
                 3)) *
            255).astype(
            np.uint8)
        self.merge_mask[10:-10, 10:-10, :] *= 0
        self.merge_mask = cv2.GaussianBlur(self.merge_mask, (21, 21), 15)
        self.merge_mask = self.merge_mask / 255

    def audio2signal(self, audio_slice: AudioSlice) -> list[SignalType]:
        len_audio = audio_slice.get_audio_duration()
        out_bs_count = int(len_audio * self._init_option.video_frame_rate)
        bs_results = []
        for _ in range(out_bs_count):
            bs_result = np.zeros((52))
            bs_results.append(bs_result)
        return bs_results

    def signal2img(self, bs_data, avatar_status: AvatarStatus) -> tuple[np.ndarray, int]:
        if avatar_status == AvatarStatus.LISTENING:
            image_data = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            image_data = np.ones((480, 640, 3), dtype=np.uint8) * 255

        return image_data, 0

    def mouth2full(self, mouth_image: np.ndarray, bg_frame_id: int, use_bg=False) -> np.ndarray:
        # mouth_image = mouth_image.numpy().astype(np.uint8)
        mouth_image = cv2.resize(mouth_image, (self.x2 - self.x1, self.y2 - self.y1))
        mouth_image = mouth_image[:, :, ::-1]
        full_img = self.bg_data_list[bg_frame_id].copy()
        if not use_bg:
            full_img[self.y1:self.y2, self.x1:self.x2, :] = mouth_image * \
                (1 - self.merge_mask) + full_img[self.y1:self.y2, self.x1:self.x2, :] * self.merge_mask
        full_img = full_img.astype(np.uint8)
        return full_img

    def get_idle_signal(self, idle_frame_count) -> list[SignalType]:
        bs_results = []
        for _ in range(idle_frame_count):
            bs_result = np.zeros((52))
            bs_results.append(bs_result)
        return bs_results

    def get_algo_config(self):
        return AvatarAlgoConfig(
            input_audio_sample_rate=16000,
            input_audio_slice_duration=1
        )
