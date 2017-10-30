#!/usr/bin/python3

import argparse
import os.path as osp
from functools import partial
from glob import glob
from multiprocessing.dummy import Pool
from subprocess import call

from tqdm import tqdm


def main():
    """Main Function"""
    # Set root dir and render script path
    root_dir = osp.dirname(osp.abspath(__file__))
    default_render_script = osp.join(root_dir, 'render_single_image_info.py')

    parser = argparse.ArgumentParser()
    parser.add_argument("image_infos_dir", nargs=1, type=str, help="Path to output directory")
    parser.add_argument("-g", "--gpus", nargs='+', default=[0], type=int, help="GPUs to use.")
    parser.add_argument("-t", "--num_of_threads", default=12, type=int, help="Number of parallel threads to use")
    parser.add_argument("-s", "--render_script", default=default_render_script, type=str, help="Path to render script")
    parser.add_argument("-e", "--extra_args", default='', type=str, help="extra arguements for the rendering script")
    parser.add_argument('--dryrun', dest='dryrun', action='store_true')
    parser.set_defaults(dryrun=False)
    args = parser.parse_args()

    image_infos_dir = args.image_infos_dir[0]
    assert osp.isdir(image_infos_dir), "{} is not a directory".format(image_infos_dir)
    assert osp.exists(image_infos_dir), "Directory {} do not exist".format(image_infos_dir)
    assert osp.exists(args.render_script), "render script {} do not exist".format(args.render_script)

    assert args.num_of_threads > 0, "Number of threads need to be +ve"
    num_of_gpus = len(args.gpus)
    assert num_of_gpus > 0, "You need to provide atleast one gpu"

    # Glob image info files
    image_info_files = glob(osp.join(image_infos_dir, '*_info.json'))
    assert image_info_files, "No image_info json files found in {}".format(image_infos_dir)

    print("Rendering will use {} gpus = {}".format(num_of_gpus, args.gpus))

    print('Generating rendering commands ...')
    commands = []
    for idx, image_info_file in enumerate(tqdm(image_info_files)):
        gpu_id = args.gpus[idx % num_of_gpus]
        command = "blender --background --python {} -- {} -g {} {} > /dev/null 2>&1".format(args.render_script, image_info_file, gpu_id, args.extra_args)
        commands.append(command)

    if args.dryrun:
        for cmd in commands:
            print(cmd)
    else:
        pool = Pool(args.num_of_threads)
        print ('Rendering images with {} threads and {} gpus ({}). This can take days.'.format(args.num_of_threads, num_of_gpus, args.gpus))
        for idx, return_code in enumerate(tqdm(pool.imap(partial(call, shell=True), commands))):
            if return_code != 0:
                print('Rendering failed for {}'.format(image_info_files[idx]))


if __name__ == '__main__':
    main()
