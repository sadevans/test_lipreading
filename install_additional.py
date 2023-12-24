import subprocess
subprocess.run(['ls'], cwd='lipreading/auto_vsr', shell=True)
subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
subprocess.run(['ls'], cwd='auto_vsr/fairseq', shell=True)
subprocess.run(['pip', 'install', '--editable', './'])
subprocess.run(['ls'], cwd='lipreading/auto_vsr', shell=True)


# subprocess.run(['ls'], cwd='lipreading/auto_vsr', shell=True)
subprocess.run(['git', 'clone', 'https://github.com/hhj1897/face_detection.git'])
subprocess.run(['ls'], cwd='auto_vsr/face_detection', shell=True)
subprocess.run(['git', 'lfs', 'pull'])

subprocess.run(['pip', 'install', '-e'])
subprocess.run(['ls'], cwd='lipreading/auto_vsr', shell=True)