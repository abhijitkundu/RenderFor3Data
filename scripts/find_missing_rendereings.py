#!/usr/bin/python3

import os.path as osp

def main():
    """Main Function"""
    for i in range(20000):
        image_file = "../data/FlyingCars20k_seed42/color_cyc/{:08d}_color.png".format(i)
        if not osp.exists(image_file):
            info_file = "../data/FlyingCars20k_seed42/image_infos/{:08d}_info.json".format(i)
            print(info_file)

if __name__ == '__main__':
    main()
