import argparse
import sys
import os.path as osp

import bpy

def main():
    """main function"""
    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='convert 3ds file to obj')
    parser.add_argument('file_3ds', type=str, help='path to 3ds file')
    args = parser.parse_args(argv)

    assert osp.exists(args.file_3ds)

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # import 3ds file
    bpy.ops.import_scene.autodesk_3ds(filepath=args.file_3ds)
    # model = bpy.context.selected_objects[0]

    # export obj file
    file_obj = osp.splitext(args.file_3ds)[0]+'.obj'
    bpy.ops.export_scene.obj(filepath=file_obj, use_selection=True)

if __name__ == '__main__':
    main()