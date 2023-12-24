import sys
import subprocess

if len(sys.argv) > 1:
    model = sys.argv[1]

    if model  == 'lipnet':
        print('Model lipnet')
        if len(sys.argv) == 4:
            print(sys.argv[3])
            subprocess.run(['python', 'lipnet/main.py', sys.argv[2], sys.argv[3]])
        else:
            print('Необходимо ввести путь к видео и к модели')
            exit
    elif model == 'auto vsr':
        print('Model auto vsr')
        if len(sys.argv) == 4:
            subprocess.run(['python', 'auto_vsr/main.py', 'data.modality=video', 'file_path=',sys.argv[2], \
                            'pretrained_model_path=',sys.argv[3]])
        elif len(sys.argv) == 5:
            subprocess.run(['python', 'auto_vsr/main.py', 'data.modality=video', 'file_path=',sys.argv[2], \
                            'anno_path=',sys.argv[3], 'pretrained_model_path=',sys.argv[4]])
        else:
            print('Необходимо ввести путь к видео и к модели')
            exit

else:
    print('Необходимо передать имя тестируемой модели')
    exit