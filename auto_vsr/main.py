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
import pandas as pd


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
        transcript = transcript.replace("'", ' ')
        print(f"TRANSCRIPT: {transcript}")

        if len(cfg.anno_path) != 0:
            wer = []
            cer = []
            lwe = []
            lce = []
            transcript_truth = torch.LongTensor([load_annotation(cfg.anno_path)]).cuda()
            truth_transcript = [arr2txt(transcript_truth[_], start=1) for _ in range(transcript_truth.size(0))]
            wer.extend(WER(transcript, truth_transcript[0])) 
            cer.extend(CER(transcript, truth_transcript[0]))
            lwe.append(LENGTH_SENTENCE_WORDS(transc, truth_transcript[0]))
            lce.append(LENGTH_SENTENCE_CHARS(transc, truth_transcript[0]))

    elif Path(cfg.file_path).is_dir():
        transcripts = []
        videos = [os.path.join(cfg.file_path, video) for video in os.listdir(cfg.file_path)]
        videos.sort()
        videos = videos[-1::-1]
        for vid in videos:
            transcript = pipeline(vid)
            transcript = transcript.replace("'", ' ')
            print(f"TRANSCRIPT: {transcript}")
            transcripts.append(transcript)
        if len(cfg.anno_path) != 0:
            wer = []
            cer = []
            lwe = []
            lce = []
            df_vsr = pd.DataFrame(columns=['video name', 'truth annotation', 'predicted annotation', \
        'len word truth', 'len word predicted', 'len char truth', 'len char predicted', 'wer', 'cer', \
        'lwer', 'lcer'])
            annotations = [os.path.join(cfg.anno_path, ann) for ann in os.listdir(cfg.anno_path)]
            annotations.sort()
            annotations = annotations[-1::-1]
            truth_annotations = []
            i_iter = 0
            for transc, ann in zip(transcripts, annotations):
                transcript_truth = torch.LongTensor([load_annotation(ann)]).cuda()
                truth_transcript = [arr2txt(transcript_truth[_], start=1) for _ in range(transcript_truth.size(0))]
                w = WER(transc, truth_transcript[0])
                c = CER(transc, truth_transcript[0])
                wer.append(np.array(WER(transc, truth_transcript[0])).mean()) 
                cer.append(np.array(CER(transc, truth_transcript[0])).mean())
                lwe.append(LENGTH_SENTENCE_WORDS(transc, truth_transcript[0]))
                lce.append(LENGTH_SENTENCE_CHARS(transc, truth_transcript[0]))

                new_row = {'video name':i_iter, 'truth annotation':truth_transcript[0], 'predicted annotation':transc, \
                'len word truth':len(truth_transcript[0].split(' ')), 'len word predicted':len(transc.split(' '))\
                , 'len char truth':len(truth_transcript[0]), 'len char predicted':len(transc), \
                'wer':np.array(WER(transc, truth_transcript[0])).mean(), 'cer':np.array(CER(transc, truth_transcript[0])).mean(), \
                'lwer':LENGTH_SENTENCE_WORDS(transc, truth_transcript[0]), 'lcer':LENGTH_SENTENCE_CHARS(transc, truth_transcript[0])

                    }
                df_vsr.loc[i_iter] = new_row
                
                i_iter += 1

            print(wer, cer)
            print(lwe, lce)
            df_vsr.to_excel('./vsr_results.xlsx', index=False)
            df_vsr.to_csv('./vsr_results.csv', index=False)



if __name__ == "__main__":
    main()