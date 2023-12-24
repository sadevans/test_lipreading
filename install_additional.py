import subprocess

subprocess.run(['pip', 'install', 'hydra-core', '--upgrade'])
subprocess.run('cd ./auto_vsr && ls', shell=True)
subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
subprocess.run('cd ./fairseq && ls', shell=True)
subprocess.run(['pip', 'install', '--editable', './'])
subprocess.run('cd .. && ls', shell=True)

subprocess.run(['git', 'clone', 'https://github.com/hhj1897/face_alignment.git'])
subprocess.run('cd ./face_alignment && ls', shell=True)
subprocess.run(['pip', 'install', '-e', '.'])
subprocess.run('cd .. && ls', shell=True)

subprocess.run(['git', 'clone', 'https://github.com/hhj1897/face_detection.git'])
subprocess.run('cd ./face_detection && ls', shell=True)
subprocess.run(['git', 'lfs', 'pull'])
subprocess.run(['pip', 'install', '-e', '.'])
subprocess.run('cd .. && ls', shell=True)
subprocess.run('cd .. && ls', shell=True)