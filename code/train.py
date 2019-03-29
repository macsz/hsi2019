import argparse
import json
import os
import sys

import numpy as np

from face import Facenet


def main(args):
    categories = sorted([d for d in os.listdir(args.cat_dir) if os.path.isdir(os.path.join(args.cat_dir, d))])
    print(categories)
    data_all = {}
    data_avg = {}
    for category in categories:
        category_path = os.path.join(args.cat_dir, category)
        print(category_path)
        fs = [os.path.join(category_path, f) for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f)) and f.split(
            '.')[-1] in ['jpg', 'png']]
        print('Files:', len(fs))
        fn = Facenet()
        images = fn.load_and_align_data(fs, 160)

        emb = fn.get_embeddings(images, args.model)
        data_all[category] = emb.tolist()

        emb_avg = np.average(emb, axis=0)
        data_avg[category] = emb_avg.tolist()

    with open('data_all.json', 'w') as outfile:
        json.dump(data_all, outfile)
    with open('data_avg.json', 'w') as outfile:
        json.dump(data_avg, outfile)

    print('Verify')
    with open('data_all.json', 'r') as outfile:
        data = json.load(outfile)
        print('Keys:', len(data.keys()))
        for key in data.keys():
            print(key, np.array(data[key]).shape)

    with open('data_avg.json', 'r') as outfile:
        data = json.load(outfile)
        print('Keys:', len(data.keys()))
        for key in data.keys():
            print(key, np.array(data[key]).shape)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('cat_dir', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
