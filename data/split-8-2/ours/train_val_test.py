import os
from random import shuffle
import subprocess


def run_cmd(cmd):
    print(cmd)
    cmd = cmd.split(' ')
    subprocess.check_output(cmd)

try:
    os.mkdir('train/')
except:
    pass
try:
    os.mkdir('val/')
except:
    pass
try:
    os.mkdir('test/')
except:
    pass


categories = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('S')])
# print(categories)

for cat in categories:
    try:
        os.mkdir('train/'+cat)
    except:
        pass
    try:
        os.mkdir('val/'+cat)
    except:
        pass
    try:
        os.mkdir('test/'+cat)
    except:
        pass

for cat in categories:
    fs = os.listdir(cat)
    # print('Files in {}: {}'.format(cat, len(fs)))
    shuffle(fs)
    print('All: {}'.format(len(fs)))
    train = sorted(fs[:len(fs)*8//10])
    val = sorted(fs[len(fs)*8//10:len(fs)*9//10])
    test = sorted(fs[len(fs)*9//10:])
    print('Train: {}'.format(len(train)))
    print('Val: {}'.format(len(val)))
    print('Test: {}'.format(len(test)))
    input(">>> Proceed?")
    for f in train:
        cmd = 'git mv {} {}'.format(cat+'/'+f, 'train/'+cat+'/'+f)
        run_cmd(cmd)
    for f in val:
        cmd = 'git mv {} {}'.format(cat+'/'+f, 'val/'+cat+'/'+f)
        run_cmd(cmd)
    for f in test:
        cmd = 'git mv {} {}'.format(cat+'/'+f, 'test/'+cat+'/'+f)
        run_cmd(cmd)
