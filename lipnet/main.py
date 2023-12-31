import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tempfile
import os
import cv2
import sys
from pathlib import Path
from dataset_inference import MyDatasetInference
import numpy as np
import face_alignment
import time
from model import LipNet
import editdistance
import torch.optim as optim
from metrics import *
import pandas as pd

import subprocess

opt = __import__('options')

letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def get_position(size, padding=0.25):
    """Function for crop lips area"""
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                    0.553364, 0.490127, 0.42689]
    
    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                    0.784792, 0.824182, 0.831803, 0.824182]
    
    x, y = np.array(x), np.array(y)
    
    x = (x + padding) / (2 * padding + 1)
    y = (y + padding) / (2 * padding + 1)
    x = x * size
    y = y * size
    return np.array(list(zip(x, y)))


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
 
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
 
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def load_video(file):
    """Load video from a specific path, extract lips patches 
    and make video from extracted lips patches"""

    p = tempfile.mkdtemp()
    cmd = 'ffmpeg -i \'{}\' -qscale:v 2 -r 25 \'{}/%d.jpg\''.format(file, p)
    # os.system(cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    files = os.listdir(p)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        
    array = [cv2.imread(os.path.join(p, file)) for file in files]
    
    
    array = list(filter(lambda im: not im is None, array))
    #array = [cv2.resize(im, (100, 50), interpolation=cv2.INTER_LANCZOS4) for im in array]
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda') # detect facial landmarks
    points = [fa.get_landmarks(I) for I in array]
    
    front256 = get_position(256)
    video = []
    for point, scene in zip(points, array):
        if(point is not None):
            shape = np.array(point[0])
            shape = shape[17:]
            M = transformation_from_points(np.matrix(shape), np.matrix(front256))
           
            img = cv2.warpAffine(scene, M[:2], (256, 256))
            (x, y) = front256[-20:].mean(0).astype(np.int32)
            w = 160//2
            img = img[y-w//2:y+w//2,x-w:x+w,...]
            img = cv2.resize(img, (128, 64))
            video.append(img)
    
    
    video = np.stack(video, axis=0).astype(np.float32)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2)) / 255.0

    return video, p


def ctc_decode(pred):
    pred = pred.argmax(-1)
    t = pred.size(0)
    result = []
    for i in range(t+1):
        result.append(MyDatasetInference.ctc_arr2txt(pred[:i], start=1))
    return result


def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)


if __name__ == '__main__':
    # opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    tic = time.time()

    model = LipNet()
    model = model.cuda()

    net = nn.DataParallel(model).cuda()
    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print('Please set pretrained weights')
        exit


    path_obj = Path(sys.argv[1])
    if path_obj.is_file():
        print('FILE')
        flag_annotation = False
        video, img_p = MyDatasetInference._load_video(sys.argv[1])

        if len(sys.argv) >=3 and Path(sys.argv[2]).is_file():
            flag_annotation = True
            annotation_truth = load_annotation(sys.argv[3])
        y_pred = model(video[None,...].cuda())

        annotation_pred = ctc_decode(y_pred[0])
        print(annotation_pred[-1])

        if flag_annotation:
            wer = []
            cer = []
            lwe = []
            lce = []
            truth_annotation = [MyDatasetInference.arr2txt(annotation_truth[_], start=1) for _ in range(annotation_truth.size(0))]
            wer.extend(WER(annotation_pred[-1], truth_annotation)) 
            cer.extend(CER(annotation_pred[-1], truth_annotation))
            lwe.append(LENGTH_SENTENCE_WORDS(annotation_pred[-1], truth_annotation[0]))
            lce.append(LENGTH_SENTENCE_CHARS(annotation_pred[-1], truth_annotation[0]))

    elif path_obj.is_dir():
        print('DIR')
        dataset = MyDatasetInference(sys.argv[1],
                sys.argv[2])
        
        loader = dataset2dataloader(dataset, shuffle=False)
        wer = []
        cer = []
        lwe = []
        lce = []
        df_lipnet = pd.DataFrame(columns=['video name', 'truth annotation', 'predicted annotation', \
        'len word truth', 'len word predicted', 'len char truth', 'len char predicted', 'wer', 'cer', \
        'lwer', 'lcer'])
        for (i_iter, input) in enumerate(loader):            
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            txt_len = input.get('txt_len').cuda()
            y_pred = model(vid)
            annotation_pred = ctc_decode(y_pred[0])
            truth_txt = [MyDatasetInference.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.append(np.array(WER(annotation_pred[-1], truth_txt[0])).mean()) 
            cer.append(np.array(CER(annotation_pred[-1], truth_txt[0])).mean())
            lwe.append(LENGTH_SENTENCE_WORDS(annotation_pred[-1], truth_txt[0]))
            lce.append(LENGTH_SENTENCE_CHARS(annotation_pred[-1], truth_txt[0]))
            new_row = {'video name':i_iter, 'truth annotation':truth_txt[0], 'predicted annotation':annotation_pred[-1], \
        'len word truth':len(truth_txt[0].split(' ')), 'len word predicted':len(annotation_pred[-1].split(' '))\
        , 'len char truth':len(truth_txt[0]), 'len char predicted':len(annotation_pred[-1]), \
        'wer':np.array(WER(annotation_pred[-1], truth_txt[0])).mean(), 'cer':np.array(CER(annotation_pred[-1], truth_txt[0])).mean(), \
        'lwer':LENGTH_SENTENCE_WORDS(annotation_pred[-1], truth_txt[0]), 'lcer':LENGTH_SENTENCE_CHARS(annotation_pred[-1], truth_txt[0])

            }
            df_lipnet.loc[i_iter] = new_row
        print(wer, cer, lwe, cer)
        print(df_lipnet)
        df_lipnet.to_excel('./lipnet_results.xlsx', index=False)
        df_lipnet.to_csv('./lipnet_results.csv', index=False)

