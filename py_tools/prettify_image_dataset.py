#!/usr/bin/python3

"""
This script reads an image dataset file and saves in more compact and prettier format.
"""

import argparse
import os.path as osp
import numpy as np
from collections import OrderedDict

from RenderFor3Data.image_dataset import ImageDataset, NoIndent


def prettify_image_dataset(input_dataset):
    pretty_dataset = ImageDataset()
    pretty_dataset.set_name(input_dataset.name())
    pretty_dataset.set_rootdir(input_dataset.rootdir())
    if 'metainfo'in input_dataset.data:
        pretty_dataset.set_metainfo(input_dataset.metainfo())
    pretty_dataset.set_image_infos(input_dataset.image_infos())
    
    for im_info in pretty_dataset.image_infos():
        for im_info_field in ['image_size', 'image_intrinsic']:
            if im_info_field in im_info:
                im_info[im_info_field] = NoIndent(im_info[im_info_field])

        for obj_info in im_info['object_infos']:
            for obj_info_field in ['bbx_visible', 'bbx_amodal', 'viewpoint', 'center_proj', 'dimension']:
                if obj_info_field in obj_info:
                    obj_info[obj_info_field] = NoIndent(np.around(obj_info[obj_info_field], decimals=6).tolist())
            for obj_info_field in ['center_dist', 'occlusion', 'truncation']:
                if obj_info_field in obj_info:
                    obj_info[obj_info_field] = float(np.around(obj_info[obj_info_field], decimals=6))
    
    return pretty_dataset


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dataset_files", nargs=1, type=str, help="Path to image dataset file to split")
    args = parser.parse_args()

    image_dataset_file = args.image_dataset_files[0]
    assert osp.exists(image_dataset_file), 'Path to image dataset file does not exist: {}'.format(image_dataset_file)

    print('Loading image dataset from {} ...'.format(image_dataset_file), end='', flush=True)
    input_dataset = ImageDataset.from_json(image_dataset_file)
    print(' Done.')

    print('Prettifying image dataset ...', end='', flush=True)
    pretty_dataset = prettify_image_dataset(input_dataset)
    print(' Done.')
    
    output_filename = "{}_prettified.json".format(pretty_dataset.name())
    pretty_dataset.write_data_to_json(output_filename)


if __name__ == '__main__':
    main()
