from fractions import Fraction
import os
import av
from loguru import logger
from src.handlers.avatar.liteavatar.avatar_output_handler import AvatarOutputHandler
from src.handlers.avatar.liteavatar.model.algo_model import AudioResult, AvatarInitOption, VideoResult, AvatarStatus
from src.engine_utils.directory_info import DirectoryInfo


class SampleOutputHandler(AvatarOutputHandler):

    def __init__(self):
        super().__init__()
        output_path = os.path.join(DirectoryInfo.get_project_dir(), 'sample_output.mp4')
        print(f"{output_path=}")
        self.output_container = av.open(output_path, mode='w')
        self.video_stream = None
        self.audio_stream = None
        self._last_audio_pts = -1
        self._init_option: AvatarInitOption = None

    def on_audio(self, audio_result: AudioResult):
        audio_res = audio_result.audio_frame.pts * float(audio_result.audio_frame.time_base)
        print(f"{audio_res=}")
        logger.info("receive audio result {:.3f}", audio_res)
        audio_frame = audio_result.audio_frame
        if self.audio_stream is None:
            self.audio_stream = self.output_container.add_stream(
                'aac', rate=self._init_option.audio_sample_rate)
            # self.audio_stream.channels = 1
            # self.audio_stream.layout = "mono"
            # self.audio_stream.time_base = Fraction(1, self._init_option.audio_sample_rate)

        assert audio_frame.pts > self._last_audio_pts
        self._last_audio_pts = audio_frame.pts
        try:
            packets = self.audio_stream.encode(audio_frame)
            for packet in packets:
                self.output_container.mux(packet)
        except Exception as e:
            logger.warning(e)

    def on_video(self, video_result: VideoResult):
        video_res = video_result.video_frame.pts * float(video_result.video_frame.time_base)
        print(f"{video_res=}")
        logger.info("receive image result {:.3f} with status {}",
                    video_res,
                    video_result.avatar_status)
        video_frame = video_result.video_frame
        if self.video_stream is None:
            self.video_stream = self.output_container.add_stream(
                'h264', rate=self._init_option.video_frame_rate)
            self.video_stream.width = video_frame.width
            self.video_stream.height = video_frame.height

        try:
            packets = self.video_stream.encode(video_frame)
            for packet in packets:
                self.output_container.mux(packet)
        except Exception as e:
            logger.warning(e)

    def on_start(self, init_option: AvatarInitOption):
        logger.info("sample handler start {}", init_option)
        self._init_option = init_option

    def on_stop(self):
        logger.info("sample handler stop")
        for packet in self.video_stream.encode():
            self.output_container.mux(packet)
        for packet in self.audio_stream.encode():
            self.output_container.mux(packet)
        self.output_container.close()
        logger.info("sample handler stopped")

    def on_avatar_status_change(self, speech_id, avatar_status: AvatarStatus):
        logger.info("speech_id {} avatar_status {}", speech_id, avatar_status)
        pass
