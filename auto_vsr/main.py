import os
import hydra

import torch
import torchaudio
import torchvision
from pathlib import Path
from datamodule.transforms import VideoTransform
from datamodule.av_dataset import cut_or_pad
from detectors.retinaface import LandmarksDetector
from detectors.video_process import VideoProcess
from lightning import ModelModule
from metrics import *


class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.landmarks_detector = LandmarksDetector(device="cuda:0")
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")
        self.modelmodule = ModelModule(cfg)
        self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        self.modelmodule.eval()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)

        with torch.no_grad():
            transcript = self.modelmodule(video)

        return transcript

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)

    if Path(cfg.file_path).is_file():
        transcript = pipeline(cfg.file_path)
        print(f"TRANSCRIPT: {transcript}")

        if len(cfg.anno_path) != 0:
            wer = []
            cer = []
            transcript_truth = torch.LongTensor([load_annotation(cfg.anno_path)]).cuda()
            truth_transcript = [arr2txt(transcript_truth[_], start=1) for _ in range(transcript_truth.size(0))]
            wer.extend(WER(transcript, truth_transcript[0])) 
            cer.extend(CER(transcript, truth_transcript[0]))
            print(np.array(wer).mean(), np.array(cer).mean())

    elif Path(cfg.file_path).is_dir():
        transcripts = []
        videos = [os.path.join(cfg.file_path, video) for video in os.listdir(cfg.file_path)]
        for vid in videos:
            transcript = pipeline(vid)
            transcripts.append(transcript)
        if len(cfg.anno_path) != 0:
            wer = []
            cer = []
            annotations = [os.path.join(cfg.anno_path, ann) for ann in os.listdir(cfg.anno_path)]
            truth_annotations = []
            for transc, ann in zip(transcripts, annotations):
                transcript_truth = torch.LongTensor([load_annotation(ann)]).cuda()
                truth_transcript = [arr2txt(transcript_truth[_], start=1) for _ in range(transcript_truth.size(0))]
                wer.extend(np.array(WER(transc, truth_transcript[0])).mean()) 
                cer.extend(np.array(CER(transc, truth_transcript[0])).mean())

            print(wer, cer)


if __name__ == "__main__":
    main()