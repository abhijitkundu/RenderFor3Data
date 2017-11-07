#!/usr/bin/python

"""
This script reads an image dataset file and saves each image info in a separate file.
"""
import argparse
import json
import os.path as osp
from glob import glob
import numpy as np
import cv2

from tqdm import tqdm

from RenderFor3Data.image_dataset import ImageDataset


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fg_image_dir", required=True, type=str, help="Path to FG image directory")
    parser.add_argument("-b", "--bg_image_dir", required=True, type=str, help="Path to BG image  directory")
    args = parser.parse_args()

    assert osp.isdir(args.fg_image_dir), '"{}" is not a directory'.format(args.fg_image_dir)
    assert osp.exists(args.fg_image_dir), '"{}" do not exist'.format(args.fg_image_dir)
    assert osp.isdir(args.bg_image_dir), '"{}" is not a directory'.format(args.bg_image_dir)
    assert osp.exists(args.bg_image_dir), '"{}" do not exist'.format(args.bg_image_dir)

    fg_image_paths = glob(osp.join(args.fg_image_dir, '*.png'))
    assert fg_image_paths, 'No FG images found'

    for fg_image_path in fg_image_paths:
        fg_image = cv2.imread(fg_image_path, cv2.IMREAD_UNCHANGED)
        assert fg_image.ndim == 3, "Image needs to be a 3D tensor"
        assert fg_image.shape[2] == 4, "FG Image needs to be a 4 channel (with alpha channel)"

        bg_image_path = osp.join(args.bg_image_dir, osp.basename(fg_image_path)[:8] + '_color.png')
        assert osp.exists(bg_image_path), 'No BG image found at {}'.format(bg_image_path)

        bg_image = cv2.imread(bg_image_path, cv2.IMREAD_COLOR)
        assert bg_image.shape[:2] == fg_image.shape[:2], 'shape mismatch b/w FG and BG'

        alpha = fg_image[:, :, 3].astype(np.float) / 255
        alpha = alpha.reshape(alpha.shape + (1,))
        fg_image = fg_image[:, :, :3].astype(np.float) / 255
        bg_image = bg_image.astype(np.float) / 255

        composite_image = alpha * fg_image + (1. - alpha) * bg_image

        cv2.namedWindow('composite_image', cv2.WINDOW_NORMAL)
        cv2.imshow('composite_image', composite_image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
