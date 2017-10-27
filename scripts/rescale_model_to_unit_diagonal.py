import argparse
import sys
import os.path as osp

import bpy
from mathutils import Matrix, Vector

# from RenderFor3Data.blender_helper import print_blender_object_atrributes

def main():
    """main function"""
    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Transform obj model')
    parser.add_argument('obj_file', type=str, help='path to obj_file file')
    args = parser.parse_args(argv)

    print("Starting to work on {}".format(args.obj_file))
    assert osp.exists(args.obj_file)

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # import obj file
    bpy.ops.import_scene.obj(filepath=args.obj_file, split_mode='OFF')
    model = bpy.context.selected_objects[0]

    # print_blender_object_atrributes(model)

    bbx_center = 0.125 * sum((Vector(b) for b in model.bound_box), Vector())
    diag_length = model.dimensions.length

    if bbx_center.length > 1e-6 or abs(diag_length - 1.0) > 1e-6:
        print("Updating model file at {}".format(args.obj_file))
        model.location = - (model.matrix_world * bbx_center)
        bpy.ops.export_scene.obj(filepath=args.obj_file, use_selection=True, global_scale=1.0/diag_length)
    else:
        print("Skipping model file at {}".format(args.obj_file))

if __name__ == '__main__':
    main()