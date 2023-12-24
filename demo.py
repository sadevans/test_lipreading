import sys
import subprocess

if len(sys.argv) > 1:
    model = sys.argv[1]

    if model  == 'lipnet':
        if len(sys.argv) == 4:
            subprocess.run(['python', 'lipnet/main.py', f'{sys.argv[2]}', f'{sys.argv[3]}'])
        else:
            print('Необходимо ввести путь к видео и к модели')
            exit
    elif model == 'auto vsr':
        print('Model auto vsr')


else:
    print('Необходимо передать имя тестируемой модели')