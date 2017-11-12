#!/usr/bin/python

"""
This script creates composite image from FG and BG image
"""

import argparse
import os.path as osp
from glob import glob
from os import makedirs
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fg_image_dir", required=True, type=str, help="Path to FG image directory")
    parser.add_argument("-o", "--out_image_dir", type=str, help="Path to output directory for Composite image ")
    parser.add_argument("-b", "--bg_file_list", required=True, type=str, help="Path to BG filelist")
    args = parser.parse_args()

    assert osp.isdir(args.fg_image_dir), '"{}" does not exist or not a directory'.format(args.fg_image_dir)
    assert osp.isfile(args.bg_file_list), '"{}" does not exist or not a file'.format(args.bg_file_list)

    with open(args.bg_file_list) as f:
        bg_image_paths = f.read().splitlines()
    print("We have {} background images".format(len(bg_image_paths)))

    for bg_image_path in bg_image_paths:
        assert osp.isfile(bg_image_path), '"{}" does not exist or not a file'.format(bg_image_path)

    shuffle(bg_image_paths)

    pause_time = 0
    if args.out_image_dir:
        if not osp.isdir(args.out_image_dir):
            makedirs(args.out_image_dir)
        pause_time = 1

    fg_image_paths = glob(osp.join(args.fg_image_dir, '*.png'))
    assert fg_image_paths, 'No FG images found'

    bg_image_index = 0
    for fg_image_path in tqdm(fg_image_paths):
        fg_image = cv2.imread(fg_image_path, cv2.IMREAD_UNCHANGED)
        assert fg_image.ndim == 3, "Image needs to be a 3D tensor"
        assert fg_image.shape[2] == 4, "FG Image needs to be a 4 channel (with alpha channel)"

        image_name = osp.basename(fg_image_path)

        if(bg_image_index == len(bg_image_paths)):
            bg_image_index = 0
            shuffle(bg_image_paths)
        bg_image_index += 1
        bg_image = cv2.imread(bg_image_paths[bg_image_index], cv2.IMREAD_COLOR)
        bg_image = cv2.resize(bg_image, (fg_image.shape[1], fg_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        assert bg_image.shape[:2] == fg_image.shape[:2], 'shape mismatch b/w FG and BG: fg_image.shape={}  bg_image.shape={}'.format(bg_image.shape,fg_image.shape)

        alpha = fg_image[:, :, 3].astype(np.float) / 255
        alpha = alpha.reshape(alpha.shape + (1,))
        fg_image = fg_image[:, :, :3].astype(np.float) / 255
        bg_image = bg_image.astype(np.float) / 255

        composite_image = (alpha * fg_image + (1. - alpha) * bg_image) * 255
        composite_image = composite_image.astype(np.uint8)

        if args.out_image_dir:
            out_image_path = osp.join(args.out_image_dir, image_name)
            cv2.imwrite(out_image_path, composite_image)

        cv2.namedWindow('composite_image', cv2.WINDOW_NORMAL)
        cv2.imshow('composite_image', composite_image)
        key = cv2.waitKey(pause_time)
        if key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
