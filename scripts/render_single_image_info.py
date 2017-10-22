"""
Blender script that takes an single image_info file and renders a scene
still image from it.
"""
import argparse
import json
import os.path as osp
import sys
import numpy as np

import bpy
from mathutils import Matrix, Vector, Color
from RenderFor3Data.blender_helper import (get_camera_intrinsic_from_blender,
                                           rotation_from_two_vectors,
                                           rotation_from_viewpoint,
                                           set_blender_camera_extrinsic,
                                           set_blender_camera_from_intrinsics,
                                           spherical_to_cartesian,
                                           print_blender_object_atrributes)


def setup_blender_engine_lights(scene):
    """Set lighting for BLENDER_RENDER engine"""
    # set environment lighting
    # bpy.context.space_data.context = 'WORLD'
    light_environment_energy_range = [0.08, 1]
    scene.world.light_settings.use_environment_light = True
    scene.world.light_settings.environment_energy = np.random.uniform(light_environment_energy_range[0], light_environment_energy_range[1])
    scene.world.light_settings.environment_color = 'PLAIN'

    light_num_range = [2, 7]
    light_dist_range = [8, 20]
    light_azimuth_range = np.radians([0.0, 360.0])
    light_elevation_range = np.radians([0.0, 90.0])
    light_energy_mean = 2
    light_energy_std = 2

    # set point lights
    for _ in range(np.random.randint(light_num_range[0], light_num_range[1])):
        light_azimuth = np.random.uniform(light_azimuth_range[0], light_azimuth_range[1])
        light_elevation = np.random.uniform(light_elevation_range[0], light_elevation_range[1])
        light_dist = np.random.uniform(light_dist_range[0], light_dist_range[1])

        # Note that For shapenet I need to use -ve directions for obj files
        lamp_location = spherical_to_cartesian(light_azimuth, light_elevation, light_dist)
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=-lamp_location)
        bpy.data.objects['Point'].data.energy = np.random.normal(light_energy_mean, light_energy_std)
        # print('lamp_location = ', lamp_location)

def setup_cycles_engine_lights():
    """Set lighting for BLENDER_RENDER engine"""

    world = bpy.data.worlds['World']
    world.use_nodes = True

    # Set background color and strength
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (np.random.uniform(0.85, 1.0), np.random.uniform(0.85, 1.0), np.random.uniform(0.85, 1.0))
    bg.inputs[1].default_value = np.random.uniform(0.06, 1.0)

    for _ in range(np.random.randint(2, 7)):
        bpy.ops.object.lamp_add(type='SUN', view_align=False, rotation=np.random.uniform(-1.0, 1.0, size=3))
        # bpy.data.objects['SUN'].data.energy = 1.5


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

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    if args.render_engine == 'CYCLES':
        # Using Cycles Render Engine
        scene.render.engine = 'CYCLES'
        bpy.data.materials['Material'].use_nodes = True
        scene.cycles.shading_system = True
        scene.use_nodes = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.cycles.film_transparent = True

        scene.cycles.device = 'GPU'
        cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = "CUDA"
        if args.gpu is not None:
            for device in cycles_prefs.devices:
                device.use = False
            assert args.gpu < len(cycles_prefs.devices), "Bad gpu provided"
            cycles_prefs.devices[args.gpu].use = True
        setup_cycles_engine_lights()
    elif args.render_engine == 'BLENDER_RENDER':
        # Using Blender Render Engine
        scene.render.engine = 'BLENDER_RENDER'
        scene.render.alpha_mode = 'TRANSPARENT'
        scene.render.use_shadows = True
        scene.render.use_raytrace = True
        setup_blender_engine_lights(scene)
    else:
        raise RuntimeError('Unhandled render engine choice made')

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

    # Set camera extrinsic to Identity
    set_blender_camera_extrinsic(cam, Matrix.Identity(3), Vector.Fill(3, 0.0))

    # Loop over all object_infos
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

    # # For debufgging save a blend file (Make sure to disable this when not debugging)
    # bpy.ops.wm.save_as_mainfile(filepath='test.blend')

    # scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = osp.basename(image_info['image_file'])
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
