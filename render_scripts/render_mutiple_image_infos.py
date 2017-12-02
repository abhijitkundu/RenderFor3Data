"""
Blender script that takes a set of single image_info files and renders them. It preloads 
all assets used in those scenes first.
"""
import argparse
import json
import os.path as osp
import sys
from datetime import datetime

import numpy as np
from tqdm import tqdm

import bpy
from mathutils import Color, Matrix, Vector
from RenderFor3Data.blender_helper import (get_camera_intrinsic_from_blender,
                                           print_blender_object_atrributes,
                                           rotation_from_two_vectors,
                                           rotation_from_viewpoint,
                                           set_blender_camera_extrinsic,
                                           set_blender_camera_from_intrinsics,
                                           spherical_to_cartesian)
from RenderFor3Data.stdout_redirector import stdout_redirector

def delete_lights():
    for obj in bpy.data.objects:
        obj.select = False
    # clear all lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)


def set_random_lights(scene, render_engine):
    """Set random lighting"""
    # delete existing lights
    delete_lights()

    # set lights based on render_engine
    if render_engine == 'CYCLES':
        world = bpy.data.worlds['World']
        world.use_nodes = True

        # Set background color and strength
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value[:3] = (np.random.uniform(0.85, 1.0), np.random.uniform(0.85, 1.0), np.random.uniform(0.85, 1.0))
        bg.inputs[1].default_value = np.random.uniform(0.06, 1.0)

        for _ in range(np.random.randint(2, 5)):
            bpy.ops.object.lamp_add(type='SUN', view_align=False, rotation=np.random.uniform(-np.pi, np.pi, size=3))
            # bpy.data.lamps['SUN'].data.energy = 1.5
    elif render_engine == 'BLENDER_RENDER':
        # Using Blender Render Engine
        # bpy.context.space_data.context = 'WORLD'
        # light_environment_energy_range = [0.08, 1]
        scene.world.light_settings.use_environment_light = True
        scene.world.light_settings.environment_energy = np.random.uniform(0.06, 1.0)
        scene.world.light_settings.environment_color = 'PLAIN'

        light_num_range = [5, 10]
        light_x_range = [-10, 10]
        light_y_range = [-2, 15]
        light_z_range = [-10, 40]

        # Add a random sun
        bpy.ops.object.lamp_add(type='SUN', view_align=False, rotation=np.random.uniform(-np.pi, np.pi, size=3))
        lamp = bpy.context.selected_objects[0]
        lamp.data.energy = np.random.uniform(0.1, 1.5)

        # set random point lights
        for _ in range(np.random.randint(light_num_range[0], light_num_range[1])):
            light_x = np.random.uniform(light_x_range[0], light_x_range[1])
            light_y = np.random.uniform(light_y_range[0], light_y_range[1])
            light_z = np.random.uniform(light_z_range[0], light_z_range[1])

            # Note that For shapenet I need to use -ve directions for obj files
            lamp_location = (light_x, light_y, light_z)
            bpy.ops.object.lamp_add(type='POINT', view_align=False, location=lamp_location)
            lamp = bpy.context.selected_objects[0]
            lamp.data.energy = np.random.uniform(0.1, 0.9)
    else:
        raise RuntimeError('Unhandled render engine choice made')


def setup_blender_engine(scene):
    """Set BLENDER_RENDER engine"""
    # Using Blender Render Engine
    scene.render.engine = 'BLENDER_RENDER'
    scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.use_shadows = True
    scene.render.use_raytrace = True

def setup_cycles_engine(scene, gpu=None):
    """Set Cycles engine"""
    # Using Cycles Render Engine
    scene.render.engine = 'CYCLES'
    # bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True
    scene.render.image_settings.color_mode = 'RGBA'
    scene.cycles.film_transparent = True

    scene.cycles.device = 'GPU'
    cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = "CUDA"
    if gpu is not None:
        for device in cycles_prefs.devices:
            device.use = False
        assert gpu < len(cycles_prefs.devices), "Bad gpu provided"
        cycles_prefs.devices[gpu].use = True


def hide_shapes(shape_files):
    """Hides all shapes"""
    for shape_name in shape_files.values():
        shape = bpy.data.objects[shape_name]
        shape.hide = True
        shape.hide_render = True

def main():
    """main function"""
    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    render_engine_choices = ['CYCLES', 'BLENDER_RENDER', 'BLENDER_GAME']
    default_dataset_rootdir = osp.join(osp.dirname(__file__), '..', 'data')

    parser = argparse.ArgumentParser(description='Render Multiple ImageInfos')
    parser.add_argument('image_info_files', type=str, nargs='+', help='path to image info files')
    parser.add_argument('-d', '--dataset_rootdir', type=str, default=default_dataset_rootdir, help='Dataset root dir')
    parser.add_argument('-r', '--render_engine', default='CYCLES', choices=render_engine_choices, help='Render engine')
    parser.add_argument('-g', '--gpu', type=int, help='GPU device number')
    parser.add_argument('--save_blend', dest='save_blend', action='store_true')
    parser.set_defaults(save_blend=False)

    args = parser.parse_args(argv)

    assert osp.exists(args.dataset_rootdir), "Dataset rootdir '{}' do not exist".format(args.dataset_rootdir)
    assert osp.isdir(args.dataset_rootdir), "Dataset rootdir '{}' is not a directory".format(args.dataset_rootdir)

    # redirect certain outputs to log file
    logfilename = 'render_mutiple_image_infos_gpu{}_uid{}_{:%Y-%m-%d_%H:%M:%S}.log'.format(args.gpu, np.random.randint(100, 1000), datetime.now())
    print("Saving log at {}".format(logfilename))
    logfile = open(logfilename, 'w')

    image_infos = []
    shape_files = {}
    print("Reading {} image_info files".format(len(args.image_info_files)))
    for image_info_file in tqdm(args.image_info_files):
        assert osp.exists(image_info_file), "File '{}' does not exist".format(image_info_file)
        with open(image_info_file, 'r') as f:
            image_info = json.load(f)
            assert 'object_infos' in image_info, "Bad image_info loaded from {}".format(image_info_file)

            current_image_shape_files = set()
            for obj_info in image_info['object_infos']:
                current_image_shape_files.add(obj_info['shape_file'])

            if len(current_image_shape_files) < len(image_info['object_infos']):
                image_name = osp.splitext(osp.basename(image_info['image_file']))[0]
                tqdm.write("Cannot handle same shape in same image. Skipping {}".format(image_name))
                continue
            for shape_file in current_image_shape_files:
                if shape_file not in shape_files:
                    shape_id = len(shape_files)
                    shape_files[shape_file] = "shape_{:04d}".format(shape_id)
            image_infos.append(image_info)
    print("We have {} unique shape files for {} images".format(len(shape_files), len(image_infos)))
    
    #Lets do the stuff we need to do only once for all the scenes

    # Setup Scene
    scene = bpy.data.scenes['Scene']

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    if args.render_engine == 'CYCLES':
        print("Using CYCLES render engine with gpu {}".format(args.gpu))
        setup_cycles_engine(scene, args.gpu)
    elif args.render_engine == 'BLENDER_RENDER':
        print("Using BLENDER render engine")
        setup_blender_engine(scene)
    else:
        raise RuntimeError('Unhandled render engine choice made')

    # set camera
    cam = bpy.data.objects['Camera']

    print("Loading shapes. This will take a long time.")
    for shape_file, shape_name in tqdm(shape_files.items()):
        # Load OBJ file
        shape_file = osp.join(args.dataset_rootdir, shape_file)
        assert osp.exists(shape_file), "Shape '{}' do not exist".format(shape_file)
        with stdout_redirector(logfile):
            bpy.ops.import_scene.obj(filepath=shape_file, split_mode='OFF')
            # bpy.ops.import_scene.obj(filepath=shape_file, axis_forward='Y', axis_up='Z')
            shape = bpy.context.selected_objects[0]
            shape.name = shape_name

    print("Rendering images. This will take a long time.")
    for image_info in tqdm(image_infos):
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

        # set random lights
        set_random_lights(scene, args.render_engine)

        # hide all shapes
        hide_shapes(shape_files)

        blender_obj_correction = Matrix(((1.0,  0.0,  0.0, 0.0),
                                         (0.0, -0.0, -1.0, 0.0),
                                         (0.0,  1.0, -0.0, 0.0),
                                         (0.0,  0.0,  0.0, 1.0)))

        # Loop over all object_infos
        for obj_info in image_info['object_infos']:
            shape_name = shape_files[obj_info['shape_file']]
            shape = bpy.data.objects[shape_name]
            shape.hide = False
            shape.hide_render = False

            # Set athe appropiate object scale
            obj_dimension = Vector(obj_info['dimension'])
            scale = obj_dimension.length / shape.dimensions.length
            mat_sca = Matrix.Scale(scale, 4)

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

            shape.matrix_world = Matrix.Translation(t) * R.to_4x4() * mat_sca * blender_obj_correction

        image_name = osp.splitext(osp.basename(image_info['image_file']))[0]
        with stdout_redirector(logfile):
            if args.save_blend:
                bpy.ops.wm.save_as_mainfile(filepath=image_name + '.blend')

            # scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = image_name + '.png'
            bpy.ops.render.render(write_still=True)

    logfile.close()


if __name__ == '__main__':
    main()
