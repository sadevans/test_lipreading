import subprocess

subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
subprocess.run(['cd', 'fairseq'])
subprocess.run(['pip', 'install', '--editable', './'])
subprocess.run(['cd', '..'])

# subprocess.run(['git', 'clone', 'https://github.com/pytorch/fairseq'])
# subprocess.run(['cd', 'fairseq'])
# subprocess.run(['pip', 'install', '--editable', './'])
# subprocess.run(['cd', '..'])
