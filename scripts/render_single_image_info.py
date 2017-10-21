import argparse
import json
import math
import os.path as osp
import sys

import numpy as np

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from RenderFor3Data.blender_helper import (
    get_camera_intrinsic_from_blender,
    get_camera_extrinsic_from_blender,
    project_by_object_utils,
    set_blender_camera_from_intrinsics,
    set_blender_camera_extrinsic,
    rotation_from_viewpoint,
    rotation_from_two_vectors,
)


def print_object_attributes(obj):
    """Helper to print object attributes"""
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))

# def init_scene(scene, camera_params, fbx_file):
#     # load fbx model
#     bpy.ops.import_scene.fbx(filepath=fbx_file, axis_forward='Y', axis_up='Z', global_scale=100)

#     obname = bpy.data.objects[4].name
#     assert '_avg' in obname

#     ob = bpy.data.objects[obname]
#     ob.data.use_auto_smooth = False  # autosmooth creates artifacts

#     # assign the existing spherical harmonics material
#     ob.active_material = bpy.data.materials['Material']

#     # delete the default cube (which held the material)
#     bpy.ops.object.select_all(action='DESELECT')
#     bpy.data.objects['Cube'].select = True
#     bpy.ops.object.delete(use_global=False)

#     # set camera properties and initial position
#     bpy.ops.object.select_all(action='DESELECT')
#     cam_ob = bpy.data.objects['Camera']
#     scn = bpy.context.scene
#     scn.objects.active = cam_ob

#     cam_ob.matrix_world = Matrix(((0., 0., 1, camera_params['camera_distance']),
#                                   (0., -1, 0., -1.0),
#                                   (-1., 0., 0., 0.),
#                                   (0.0, 0.0, 0.0, 1.0)))
#     cam_ob.data.angle = math.radians(40)
#     cam_ob.data.lens = 60
#     cam_ob.data.clip_start = 0.1
#     cam_ob.data.sensor_width = 32

#     # setup an empty object in the center which will be the parent of the Camera
#     # this allows to easily rotate an object around the origin
#     scn.cycles.film_transparent = True
#     scn.render.layers["RenderLayer"].use_pass_vector = True
#     scn.render.layers["RenderLayer"].use_pass_normal = True
#     scene.render.layers['RenderLayer'].use_pass_emit = True
#     scene.render.layers['RenderLayer'].use_pass_emit = True
#     scene.render.layers['RenderLayer'].use_pass_material_index = True

#     # set render size
#     scn.render.resolution_x = camera_params['resy']
#     scn.render.resolution_y = camera_params['resx']
#     scn.render.resolution_percentage = 100
#     scn.render.image_settings.file_format = 'PNG'

#     # clear existing animation data
#     ob.data.shape_keys.animation_data_clear()
#     arm_ob = bpy.data.objects['Armature']
#     arm_ob.animation_data_clear()

#     return(ob, obname, arm_ob, cam_ob)

# # transformation between pose and blendshapes


def main():
    """main function"""
    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    render_engine_choices = ['CYCLES', 'BLENDER_RENDER', 'BLENDER_GAME']

    parser = argparse.ArgumentParser(description='Render Single ImageInfo')
    parser.add_argument('image_info_file', type=str, nargs=1, help='path to image info file')
    parser.add_argument('-r', '--render_engine', default='CYCLES', choices=render_engine_choices, help='Render engine')

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
        scene.cycles.device = 'GPU'
        bpy.data.materials['Material'].use_nodes = True
        scene.cycles.shading_system = True
        # scene.use_nodes = True
        scene.render.image_settings.color_mode = 'RGBA'
        scene.cycles.film_transparent = True
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
