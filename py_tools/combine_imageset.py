#!/usr/bin/python

"""
This script selects images from multiple image dirs into one
"""
import argparse
import os.path as osp
from os import makedirs
from glob import glob
import numpy as np
from shutil import copy2
from tqdm import tqdm


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dirs", nargs='+', required=True, type=str, help="Path to list of input directories")
    parser.add_argument("-p", "--probs", nargs='+', type=float, help="Probability of selecting from each input dir")
    parser.add_argument("-o", "--out_dir", required=True, type=str, help="Path to output directory")
    args = parser.parse_args()

    num_of_dirs = len(args.input_dirs)
    assert num_of_dirs, 'num of input dirs needs to be +ve'

    if args.probs:
        assert len(args.probs) == num_of_dirs
        probs = np.array(args.probs)
        probs = probs / probs.sum()
    else:
        probs = np.full((num_of_dirs,), 1. / num_of_dirs)

    if not osp.isdir(args.out_dir):
        makedirs(args.out_dir)

    image_lists = []
    for input_dir in args.input_dirs:
        assert osp.isdir(input_dir), '"{}" does not exist or not a directory'.format(input_dir)
        image_list = glob(osp.join(input_dir, '*.png'))
        assert image_list, 'No images found in {}'.format(input_dir)
        image_lists.append(image_list)

    num_of_images_per_dir = len(image_lists[0])
    assert all(len(i) == num_of_images_per_dir for i in image_lists)
    print("We have {} images in each of the {} directories".format(num_of_images_per_dir, num_of_dirs))

    for image_paths in tqdm(zip(*image_lists), total=num_of_images_per_dir):
        image_path = np.random.choice(image_paths, p=probs)
        copy2(image_path, args.out_dir)



if __name__ == '__main__':
    main()
