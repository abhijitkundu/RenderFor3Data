import argparse
import math
import os.path as osp
import sys
import bpy
from mathutils import Matrix, Vector
from RenderFor3Data.blender_helper import rotation_from_viewpoint, set_blender_camera_extrinsic


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

    # Setup Scene
    scene = bpy.data.scenes['Scene']

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # import obj file
    bpy.ops.import_scene.obj(filepath=args.obj_file, split_mode='OFF')
    # model = bpy.context.selected_objects[0]

    scene.render.engine = 'BLENDER_RENDER'
    # scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.use_shadows = True
    scene.render.use_raytrace = True
    scene.world.light_settings.use_environment_light = True
    scene.world.light_settings.environment_energy = 0.4
    scene.world.light_settings.environment_color = 'PLAIN'

    bpy.ops.object.lamp_add(type='SUN')

    cam = bpy.data.objects['Camera']
    viewpoint = Vector((-45, 25., 0.0)) * math.pi / 180.0
    set_blender_camera_extrinsic(cam, rotation_from_viewpoint(viewpoint), Vector((0., 0., 1.5)))

    scene.render.resolution_x = 480
    scene.render.resolution_y = 270
    scene.render.resolution_percentage = 100

    # scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = osp.splitext(args.obj_file)[0] + '.png'
    bpy.ops.render.render(write_still=True)
    print('Saved rendered scene at {}'.format(scene.render.filepath))



if __name__ == '__main__':
    main()
