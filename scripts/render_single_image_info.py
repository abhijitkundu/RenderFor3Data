"""
Blender script that takes an single image_info file and renders a scene
still image from it.
"""
import argparse
import json
import os.path as osp
import sys

import bpy
from mathutils import Matrix, Vector
from RenderFor3Data.blender_helper import (get_camera_intrinsic_from_blender,
                                           rotation_from_two_vectors,
                                           rotation_from_viewpoint,
                                           set_blender_camera_extrinsic,
                                           set_blender_camera_from_intrinsics)


def main():
    """main function"""
    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    render_engine_choices = ['CYCLES', 'BLENDER_RENDER', 'BLENDER_GAME']

    parser = argparse.ArgumentParser(description='Render Single ImageInfo')
    parser.add_argument('image_info_file', type=str, nargs=1, help='path to image info file')
    parser.add_argument('-r', '--render_engine', default='CYCLES', choices=render_engine_choices, help='Render engine')
    parser.add_argument('-g', '--gpu', type=int, help='GPU device number')

    args = parser.parse_args(argv)
    image_info_file = args.image_info_file[0]

    assert osp.exists(image_info_file), "File '{}' does not exist".format(image_info_file)

    with open(image_info_file, 'r') as f:
        image_info = json.load(f)

    # Setup Scene
    scene = bpy.data.scenes['Scene']

    if args.render_engine == 'CYCLES':
        # Using Cycles Render Engine
        scene.render.engine = 'CYCLES'
        bpy.data.materials['Material'].use_nodes = True
        scene.cycles.shading_system = True
        # scene.use_nodes = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.cycles.film_transparent = True

        scene.cycles.device = 'GPU'
        cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = "CUDA"
        if args.gpu:
            for device in cycles_prefs.devices:
                device.use = False
            assert args.gpu < len(cycles_prefs.devices), "Bad gpu provided"
            cycles_prefs.devices[args.gpu].use = True
    elif args.render_engine == 'BLENDER_RENDER':
        # Using Blender Render Engine
        scene.render.engine = 'BLENDER_RENDER'
        scene.render.alpha_mode = 'TRANSPARENT'
        # scene.render.use_shadows = False
        # scene.render.use_raytrace = False
    else:
        raise RuntimeError('Unhandled render engine choice made')

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    # clear default lights
    # bpy.ops.object.select_by_type(type='LAMP')
    # bpy.ops.object.delete(use_global=False)

    cam = bpy.data.objects['Camera']

    W = image_info['image_size'][0]
    H = image_info['image_size'][1]

    # set render size
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    set_blender_camera_from_intrinsics(cam, image_info['image_intrinsic'])

    K = get_camera_intrinsic_from_blender(cam)
    Kinv = K.inverted()

    set_blender_camera_extrinsic(cam, Matrix.Identity(3), Vector.Fill(3, 0.0))

    for obj_info in image_info['object_infos']:
        model_name = "model_{:02d}".format(obj_info['id'])
        # Load OBJ file
        bpy.ops.import_scene.obj(filepath=obj_info['shape_file'], split_mode='OFF')
        # bpy.ops.import_scene.obj(filepath=image_info['shape_file'], axis_forward='Y', axis_up='Z')
        model = bpy.context.selected_objects[0]
        model.name = model_name

        # Compute ray through object center
        center_proj = obj_info['center_proj']
        center_proj_ray = Kinv * Vector((center_proj[0], center_proj[1], 1.0))

        # Rotation to account for object not along principal axes
        Rdelta = rotation_from_two_vectors(Vector((0., 0., 1.)), center_proj_ray)

        # rotation by viewwpoint
        Rvp = rotation_from_viewpoint(obj_info['viewpoint'])

        # Compute final object pose as R|t = R_delta * [R_vp| t_vp]
        R = Rdelta * Rvp
        t = Rdelta * Vector((0., 0., obj_info['center_dist']))

        model.matrix_world = Matrix.Translation(t) * R.to_4x4() * model.matrix_world

    # scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = osp.basename(image_info['image_file'])
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
