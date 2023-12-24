from setuptools import setup

setup(
    name = 'lipreading',
    version = '1.0',
    install_requires = [
        'torch==2.1.0',
        'face-alignment==1.1.1',
        'opencv-python>=3.4.2',

        'pytorch-lightning==1.5.10',
        'sentencepiece',
        'av',
        'hydra-core --upgrade'
    ],
    entry_points = {
        'console_scripts' : [
            'install_additional = lipreading.install_additional:install_additional'
        ],
    }

)