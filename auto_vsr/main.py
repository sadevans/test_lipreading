import os
import hydra

import torch
import torchaudio
import torchvision
from datamodule.transforms import VideoTransform
from datamodule.av_dataset import cut_or_pad
from detectors.retinaface import LandmarksDetector
from detectors.video_process import VideoProcess
from lightning import ModelModule



class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        # if self.modality in ["audio", "audiovisual"]:
        #     self.audio_transform = AudioTransform(subset="test")
        # if self.modality in ["video", "audiovisual"]:
            # if detector == "mediapipe":
            #     from preparation.detectors.mediapipe.detector import LandmarksDetector
            #     from preparation.detectors.mediapipe.video_process import VideoProcess
            #     self.landmarks_detector = LandmarksDetector()
            #     self.video_process = VideoProcess(convert_gray=False)
            # elif detector == "retinaface":
                # from preparation.detectors.retinaface.detector import LandmarksDetector
                # from preparation.detectors.retinaface.video_process import VideoProcess
        self.landmarks_detector = LandmarksDetector(device="cuda:0")
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        # if cfg.data.modality in ["audio", "visual"]:
        #     from lightning import ModelModule
        # elif cfg.data.modality == "audiovisual":
        #     from lightning_av import ModelModule
        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        self.modelmodule.eval()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."

        # if self.modality in ["audio", "audiovisual"]:
        #     audio, sample_rate = self.load_audio(data_filename)
        #     audio = self.audio_process(audio, sample_rate)
        #     audio = audio.transpose(1, 0)
        #     audio = self.audio_transform(audio)

        # if self.modality in ["video", "audiovisual"]:
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)

        # if self.modality in ["audio", "video"]:
        with torch.no_grad():
            transcript = self.modelmodule(video)

        return transcript

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.file_path)
    print(f"transcript: {transcript}")


if __name__ == "__main__":
    main()