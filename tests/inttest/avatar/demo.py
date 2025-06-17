
import os
import sys
import time

from loguru import logger
from src.handlers.avatar.liteavatar.avatar_processor_factory import AvatarAlgoType, AvatarProcessorFactory
from src.handlers.avatar.liteavatar.model.algo_model import AvatarInitOption
from src.handlers.avatar.liteavatar.model.audio_input import SpeechAudio
from src.engine_utils.directory_info import DirectoryInfo
from src.engine_utils.media_utils import AudioUtils
from tests.inttest.avatar.sample_output_handler import SampleOutputHandler


class AvatarDemo:
    def __init__(self):
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    def run(self):
        # https://products.aspose.app/audio/zh-cn/voice-recorder/wav
        test_input_file_path = os.path.join(
            DirectoryInfo.get_project_dir(), "resource", "audio", "asr_example_zh.wav"
        )
        audio_bytes, sample_rate = AudioUtils.read_wav_to_bytes(test_input_file_path)

        processor = AvatarProcessorFactory.create_avatar_processor(
            None,
            AvatarAlgoType.TTS2FACE_CPU,
            # AvatarAlgoType.SAMPLE,
            AvatarInitOption(
                audio_sample_rate=sample_rate,
                video_frame_rate=25,
                avatar_name="20250408/sample_data",
                debug=True,
                enable_fast_mode=False,
                use_gpu=False,
            ))
        processor.register_output_handler(SampleOutputHandler())
        processor.start()

        time.sleep(1)
        processor.add_audio(SpeechAudio(
            audio_data=audio_bytes,
            speech_id="1",
            end_of_speech=True,
            sample_rate=sample_rate))
        time.sleep(20)

        print("---" * 20)
        test_input_file_path = os.path.join(
            DirectoryInfo.get_project_dir(), "resource", "audio", "asr_example_zh.wav"
        )
        audio_bytes, sample_rate = AudioUtils.read_wav_to_bytes(test_input_file_path)
        processor.add_audio(SpeechAudio(
            audio_data=audio_bytes,
            speech_id="2",
            end_of_speech=True,
            sample_rate=sample_rate))
        time.sleep(40)

        processor.stop()


if __name__ == "__main__":
    demo = AvatarDemo()
    demo.run()
