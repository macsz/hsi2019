import os
from random import shuffle
import subprocess


def run_cmd(cmd):
    print(cmd)
    cmd = cmd.split(' ')
    subprocess.check_output(cmd)


os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)


categories = sorted(set([d.split('_')[0] for d in os.listdir('.') if os.path.isfile(d)
                         and d not in ['train', 'test', 'val']]))
categories = [c for c in categories if c not in ['train', 'test', 'val']]

for cat in categories:
    os.makedirs('train/'+cat, exist_ok=True)
    os.makedirs('val/'+cat, exist_ok=True)
    os.makedirs('test/'+cat, exist_ok=True)

total_fs = 0
for cat in categories:
    fs = [f for f in os.listdir('.') if f.startswith(cat) and os.path.isfile(f) and f.endswith('jpg')]
    print('Cat:', cat, len(fs))
    total_fs += len(fs)
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
        cmd = 'git mv {} {}'.format(f, 'train/'+cat+'/'+f)
        run_cmd(cmd)
    for f in val:
        cmd = 'git mv {} {}'.format(f, 'val/'+cat+'/'+f)
        run_cmd(cmd)
    for f in test:
        cmd = 'git mv {} {}'.format(f, 'test/'+cat+'/'+f)
        run_cmd(cmd)
print('Total fs:', total_fs)
