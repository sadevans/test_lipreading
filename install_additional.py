import subprocess
import os

subprocess.run(['pip', 'install', 'hydra-core', '--upgrade'])

if not os.path.exists('./auto_vsr/fairseq'):
    os.chdir('auto_vsr')
    subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
    os.chdir('fairseq')
    subprocess.run(['pip', 'install', '--editable', './'])
    os.chdir('..')
if not os.path.exists('./auto_vsr/face_alignment'):

    subprocess.run(['git', 'clone', 'https://github.com/hhj1897/face_alignment.git'])
    os.chdir('face_alignment')
    subprocess.run(['pip', 'install', '-e', '.'])
    os.chdir('..')

if not os.path.exists('./auto_vsr/face_detection'):
    subprocess.run(['git', 'clone', 'https://github.com/hhj1897/face_detection.git'])
    os.chdir('face_detection')

    subprocess.run(['git', 'lfs', 'pull'])
    subprocess.run(['pip', 'install', '-e', '.'])
    os.chdir('..')
    os.chdir('..')