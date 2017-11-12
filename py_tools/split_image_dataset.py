#!/usr/bin/python3

"""
This script splits an image dataset with a split file
"""


import argparse
import json
import os.path as osp
from collections import OrderedDict

from tqdm import tqdm

from RenderFor3Data.image_dataset import ImageDataset, NoIndent


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image_dataset_file", required=True, type=str, help="Path to image dataset file to split")
    parser.add_argument("-s", "--split_file", required=True, type=str, help="Path to output directory")
    args = parser.parse_args()

    assert osp.isfile(args.split_file), '{} either do not exist or not a file'.format(args.split_file)
    assert osp.isfile(args.image_dataset_file), '{} either do not exist or not a file'.format(args.image_dataset_file)

    split_name = osp.splitext(osp.basename(args.split_file))[0]
    with open(args.split_file) as f:
        image_names = set(f.read().splitlines())
    
    print('Found {} images in split {}'.format(len(image_names), split_name))
    print('Loading image dataset from {} ...'.format(args.image_dataset_file), end='', flush=True)
    image_datset = ImageDataset.from_json(args.image_dataset_file)
    print(' Done.')
    print(image_datset)

    split_image_infos = []

    print('Making split dataset')
    for im_info in tqdm(image_datset.image_infos()):
        image_name = osp.splitext(osp.basename(im_info['image_file']))[0]
        if image_name in image_names:
            new_im_info = OrderedDict()

            for im_info_field in ['image_file', 'segm_file']:
                if im_info_field in im_info:
                    new_im_info[im_info_field] = im_info[im_info_field]
            
            for im_info_field in ['image_size', 'image_intrinsic']:
                if im_info_field in im_info:
                    new_im_info[im_info_field] = NoIndent(im_info[im_info_field])

            new_obj_infos = []
            for obj_info in im_info['object_infos']:
                new_obj_info = OrderedDict()

                for obj_info_field in ['id', 'category']:
                    if obj_info_field in obj_info:
                        new_obj_info[obj_info_field] = obj_info[obj_info_field]

                for obj_info_field in ['viewpoint', 'bbx_visible', 'bbx_amodal', 'center_proj', 'dimension']:
                    if obj_info_field in obj_info:
                        new_obj_info[obj_info_field] = NoIndent(obj_info[obj_info_field])
                
                for obj_info_field in ['center_dist', 'occlusion', 'truncation', 'shape_file']:
                    if obj_info_field in obj_info:
                        new_obj_info[obj_info_field] = obj_info[obj_info_field]
                
                new_obj_infos.append(new_obj_info)
            
            new_im_info['object_infos'] = new_obj_infos
            split_image_infos.append(new_im_info)

    split_dataset_name = "{}_{}".format(image_datset.name(), split_name)
    image_datset.set_image_infos(split_image_infos)
    image_datset.set_name(split_dataset_name)
    print(image_datset)

    image_datset.write_data_to_json(split_dataset_name + ".json")







if __name__ == '__main__':
    main()
