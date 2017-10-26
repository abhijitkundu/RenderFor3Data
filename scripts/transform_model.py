import argparse
import os.path as osp
import sys
import bpy
from mathutils import Matrix, Vector


def read_transform_file(transform_file):
    with open(transform_file) as f:
        for line in f:
            tokens = line.rstrip('\n').split()
            if tokens[0] == 'Transform':
                tokens = tokens[2:]
                assert len(tokens) == 16
                assert float(tokens[12]) == 0
                assert float(tokens[13]) == 0
                assert float(tokens[14]) == 0
                assert float(tokens[15]) == 1
                T = Matrix(((float(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3])),
                            (float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7])),
                            (float(tokens[8]), float(tokens[9]),float(tokens[10]),float(tokens[11])),
                            (0,                               0,                0,               1)))
                return T


def main():
    """main function"""
    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Transform obj model')
    parser.add_argument('obj_file', type=str, help='path to obj_file file')
    args = parser.parse_args(argv)

    model_name = osp.splitext(osp.basename(args.obj_file))[0]
    transform_file = osp.join(osp.dirname(args.obj_file), model_name + '.ini')

    assert osp.exists(transform_file), "{} does not exist".format(transform_file)
    assert osp.exists(args.obj_file), "{} does not exist".format(args.obj_file)

    T = read_transform_file(transform_file)

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # import obj file
    bpy.ops.import_scene.obj(filepath=args.obj_file, split_mode='OFF')
    model = bpy.context.selected_objects[0]

    # Apply transformation
    model.matrix_world = T * model.matrix_world

    # export obj file
    bpy.ops.export_scene.obj(filepath=args.obj_file, use_selection=True)


if __name__ == '__main__':
    main()
