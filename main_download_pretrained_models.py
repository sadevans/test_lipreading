import argparse
import os
import requests
import re
import gdown
import sys


def download_pretrained_model(model, model_name='dncnn3.pth'):
    model_dir = os.path.join(model, 'pretrained')
    if os.path.exists(os.path.join(model_dir, model_name)):
        print(f'already exists, skip downloading [{model_name}]')
    else:
        os.makedirs(model_dir, exist_ok=True)
        if model_name == 'vsr_trlrs3_23h_base.pth':
            url = 'https://drive.google.com/file/d/1OBEHbStKKFG7VDij14RDLN9VYSdE_Bhs'
        elif model_name == 'vsr_trlrs3_base.pth':
            url = 'https://drive.google.com/file/d/1aawSjxIL2ewo0W0fg4TBQgR8WMAmPeSL'
    
        elif model_name == 'vsr_trlrs3vox2_base.pth':
            url = 'https://drive.google.com/file/d/1mLAuCnK2y7zbmiHlAXMqPSF_ApGqfbAD'
        elif model_name == 'vsr_trlrwlrs2lrs3vox2avsp_base.pth': 
            url = 'https://drive.google.com/file/d/19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi'   

    gdown.download(url, model_dir, quite=True)


if __name__ == '__main__':
    download_pretrained_model(sys.argv[1], sys.argv[2])

    # r = requests.get(url, allow_redirects=True)
    # print(f'downloading [{model_dir}/{model_name}] ...')
    # open(os.path.join(model_dir, model_name), 'wb').write(r.content)
    # print('done!')