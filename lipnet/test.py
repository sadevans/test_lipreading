import os
import hydra

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchaudio
import torchvision
from auto_vsr.datamodule.transforms import VideoTransform
from auto_vsr.datamodule.av_dataset import cut_or_pad
from auto_vsr.detectors.retinaface import LandmarksDetector
from auto_vsr.detectors.video_process import VideoProcess
from auto_vsr.lightning import ModelModule
from model import LipNet
letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

opt = __import__('options')

class InferencePipeline(torch.nn.Module):
    def __init__(self, cfg, detector="retinaface"):
        super(InferencePipeline, self).__init__()
        self.modality = cfg.data.modality
        self.landmarks_detector = LandmarksDetector(device="cuda:0")
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")
        # self.modelmodule = ModelModule(cfg)

        self.model = LipNet()
        self.model = self.model.cuda()

        self.net = nn.DataParallel(self.model).cuda()

        pretrained_dict = torch.load(opt.weights)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


        # self.modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        # self.modelmodule.eval()


    def forward(self, data_filename):
        data_filename = os.path.abspath(data_filename)
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.FloatTensor(video)
        # vide
        video = video.permute((0, 3, 1, 2))
        video = self.video_transform(video)

        with torch.no_grad():
            # transcript = self.modelmodule(video)
            transcript = self.model(video.cuda())

        return transcript

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()


@staticmethod
def ctc_arr2txt(arr, start):
    pre = -1
    txt = []
    for n in arr:
        if(pre != n and n >= start):                
            if(len(txt) > 0 and txt[-1] == ' ' and letters[n - start] == ' '):
                pass
            else:
                txt.append(letters[n - start])                
        pre = n
    return ''.join(txt).strip()

def ctc_decode(pred):
    pred = pred.argmax(-1)
    t = pred.size(0)
    result = []
    for i in range(t+1):
        result.append(ctc_arr2txt(pred[:i], start=1))
    return result

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    pipeline = InferencePipeline(cfg)
    transcript = pipeline(cfg.file_path)
    annotation_pred = ctc_decode(transcript[0])
    print(f"transcript: {annotation_pred[-1]}")


if __name__ == "__main__":
    main()