import subprocess

# def install_additional():
#     subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
#     subprocess.run(['cd', 'fairseq'])
#     subprocess.run(['pip', 'install', '--editable', './'])
#     subprocess.run(['cd', '..'])

subprocess.run(['ls'], cwd='auto_vsr', shell=True)
subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
# subprocess.run(['%cd', 'fairseq'])
subprocess.run(['ls'], cwd='fairseq', shell=True)

subprocess.run(['pip', 'install', '--editable', './'])
# subprocess.run(['%cd', '..'])
subprocess.run(['ls'], cwd='auto_vsr', shell=True)

# subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
# subprocess.run(['cd', 'fairseq'])
# subprocess.run(['pip', 'install', '--editable', './'])
# subprocess.run(['cd', '..'])
