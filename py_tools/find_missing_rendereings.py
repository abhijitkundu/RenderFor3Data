#!/usr/bin/python3

import os.path as osp
import argparse

def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="Path to image dir")
    parser.add_argument("-n", "--num_of_images", default=20000, type=int, help="Number of expected images")
    args = parser.parse_args()

    assert osp.isdir(args.image_dir), "{} does not exist or not a directory".format(args.image_dir)

    for i in range(args.num_of_images):
        image_file = osp.join(args.image_dir, "{:08d}_color.png".format(i))
        if not osp.exists(image_file):
            info_file = "{:08d}_info.json".format(i)
            print(info_file)

if __name__ == '__main__':
    main()
