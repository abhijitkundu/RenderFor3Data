#!/usr/bin/python3

"""
This script reads an image dataset file and saves each image info in a separate file.
"""


import argparse
import json
import os.path as osp

from tqdm import tqdm

from RenderFor3Data.image_dataset import ImageDataset, NoIndent, DatasetJSONEncoder


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", required=True, type=str, help="Path to output directory")
    parser.add_argument("-d", "--image_dataset_file", required=True, type=str, help="Path to image dataset file to split")
    args = parser.parse_args()

    print("------------- Config ------------------")
    for arg in vars(args):
        print("{} \t= {}".format(arg, getattr(args, arg)))

    assert osp.isdir(args.out_dir), 'Output dir "{}" is not a directory'.format(args.out_dir)
    assert osp.exists(args.out_dir), 'Output dir "{}" does not exist'.format(args.out_dir)
    assert osp.exists(args.image_dataset_file), 'Path to image dataset file does not exist: {}'.format(args.image_dataset_file)

    print('Loading image dataset from {} ...'.format(args.image_dataset_file), end='', flush=True)
    image_datset = ImageDataset.from_json(args.image_dataset_file)
    print(' Done.')
    print(image_datset)

    print('Saving image infos. May take a long time')
    for image_info in tqdm(image_datset.image_infos()):
        for image_info_field in ['image_size', 'image_intrinsic']:
            if image_info_field in image_info:
                image_info[image_info_field] = NoIndent(image_info[image_info_field])

        for obj_info in image_info['object_infos']:
            shape_name = osp.splitext(osp.basename(obj_info['shape_file']))[0]
            # TODO use synset mapping of car --> 02958343 rather than hardcoding
            shape_file = osp.join(image_datset.rootdir(), 'ShapeNetCore_v1', '02958343', shape_name, 'model.obj')
            assert osp.exists(shape_file), "Shape file {} do not exist".format(shape_file)
            obj_info['shape_file'] = shape_file
            for obj_info_field in ['bbx_visible', 'bbx_amodal', 'viewpoint', 'center_proj', 'dimension']:
                if obj_info_field in obj_info:
                    obj_info[obj_info_field] = NoIndent(obj_info[obj_info_field])


        image_file_name = osp.splitext(osp.basename(image_info['image_file']))[0]
        output_filepath = osp.join(args.out_dir, '{}.json'.format(image_file_name.replace('color', 'info')))
        with open(output_filepath, 'w') as f:
            json.dump(image_info, f, indent=2, separators=(',', ':'), cls=DatasetJSONEncoder)


if __name__ == '__main__':
    main()
