import argparse
import os
import requests
import re


def download_pretrained_model(model, model_name='dncnn3.pth'):
    model_dir = os.path.join(model, 'pretrained')
    if os.path.exists(os.path.join(model_dir, model_name)):
        print(f'already exists, skip downloading [{model_name}]')
    else:
        os.makedirs(model_dir, exist_ok=True)
        if model_name == 'vsr_trlrs3_23h_base.pth':
            url = 'https://drive.usercontent.google.com/download?id=1OBEHbStKKFG7VDij14RDLN9VYSdE_Bhs&export=download&authuser=0'

        # if 'SwinIR' in model_name:
        #     url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(model_name)
        # elif 'VRT' in model_name:
        #     url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/{}'.format(model_name)
        # else:
        #     url = 'https://github.com/cszn/KAIR/releases/download/v1.0/{}'.format(model_name)
        r = requests.get(url, allow_redirects=True)
        print(f'downloading [{model_dir}/{model_name}] ...')
        open(os.path.join(model_dir, model_name), 'wb').write(r.content)
        print('done!')