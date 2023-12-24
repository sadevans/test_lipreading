import editdistance
import numpy as np

letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


@staticmethod
def txt2arr(txt, start):
    arr = []
    for c in list(txt):
        arr.append(letters.index(c) + start)
    return np.array(arr)


@staticmethod
def arr2txt(arr, start):
    txt = []
    for n in arr:
        if(n >= start):
            txt.append(letters[n - start])     
    return ''.join(txt).strip()


def load_annotation(name):
    with open(name, 'r') as f:
        lines = [line.strip().split(' ') for line in f.readlines()]
        txt = [line[2] for line in lines]
        txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))

    return txt2arr(' '.join(txt).upper(), 1)

@staticmethod
def WER(predict, truth):
    """Word Error Rate"""  
    word_pairs = [(p[0], p[1]) for p in zip(predict.split(' '), truth.split(' '))]
    wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
    return wer
    

@staticmethod
def CER(predict, truth):
    """Character Error Rate"""  
    cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
    return cer

@staticmethod
def LENGTH_SENTENCE_WORDS(predict, truth):
    """Length in words error"""
    # cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
    tr = len(truth.split(' '))
    pr = len(predict.split(' '))
    return 1 - pr/tr

@staticmethod
def LENGTH_SENTENCE_CHARS(predict, truth):
    """Length in words error"""
    # cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
    tr = len(truth)
    pr = len(predict)
    return 1 - pr/tr