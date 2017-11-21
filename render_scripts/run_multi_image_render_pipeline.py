#!/usr/bin/python3

import argparse
import os.path as osp
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

def main():
    """Main Function"""
    # Set root dir and render script path
    root_dir = osp.dirname(osp.abspath(__file__))
    default_render_script = osp.join(root_dir, 'render_mutiple_image_infos.py')

    parser = argparse.ArgumentParser()
    parser.add_argument("image_info_files", nargs='+', type=str, help="Path to image_info files")
    parser.add_argument("-g", "--gpus", nargs='+', default=[0], type=int, help="GPUs to use.")
    parser.add_argument("-t", "--num_of_threads", default=12, type=int, help="Number of parallel threads to use")
    parser.add_argument("-s", "--render_script", default=default_render_script, type=str, help="Path to render script")
    parser.add_argument("-e", "--extra_args", default='', type=str, help="extra arguements for the rendering script")
    parser.add_argument('--dryrun', dest='dryrun', action='store_true')
    parser.set_defaults(dryrun=False)
    args = parser.parse_args()

    assert osp.exists(args.render_script), "render script {} do not exist".format(args.render_script)

    assert args.num_of_threads > 0, "Number of threads need to be +ve"
    num_of_gpus = len(args.gpus)
    assert num_of_gpus > 0, "You need to provide atleast one gpu"

    print("Rendering will use {} gpus = {}".format(num_of_gpus, args.gpus))

    print('Generating rendering commands for {} images ...'.format(len(args.image_info_files)))
    commands = []
    for t in range(args.num_of_threads):
        gpu_id = args.gpus[t % num_of_gpus]
        command = "blender --background --python {} -- -g {} {}".format(args.render_script, gpu_id, args.extra_args)
        commands.append(command)
    
    assert len(commands) == args.num_of_threads

    for idx, image_info_file in enumerate(args.image_info_files):
        assert osp.exists(image_info_file), "{} do not exist".format(image_info_file)
        assert osp.isfile(image_info_file), "{} is not a file".format(image_info_file)
        assert osp.splitext(image_info_file)[1] == '.json', "File {} is not avalid json file".format(image_info_file)

        thread_id = idx % args.num_of_threads
        commands[thread_id] += " {}".format(image_info_file)

    if args.dryrun:
        for cmd in commands:
            print("-------------------------------------------------------------------")
            print(cmd)
            print("-------------------------------------------------------------------")
    else:
        pool = Pool(args.num_of_threads)
        print ('Rendering images with {} threads and {} gpus ({}). This can take days.'.format(args.num_of_threads, num_of_gpus, args.gpus))
        for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
            if return_code != 0:
                print('Rendering failed for {}'.format(args.image_info_files[idx]))


if __name__ == '__main__':
    main()
