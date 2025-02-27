"""
blender render script for smpl bodies
"""
import argparse
import os.path as osp
import sys
import json
from pickle import load
from shutil import copyfile

import numpy as np

import bpy
from mathutils import Matrix, Vector
from RenderFor3Data.blender_helper import (get_camera_intrinsic_from_blender,
                                           rotation_from_two_vectors,
                                           rotation_from_viewpoint,
                                           set_blender_camera_extrinsic,
                                           set_blender_camera_from_intrinsics,
                                           set_blender_object_hide)
from RenderFor3Data.smpl_helper import (SMPLBody, 
                                        create_shader_material,
                                        load_smpl_fbx_files)


def setup_scene(scene):
    """
    Clear default cube and lamp
    Set Render engine
    """
    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    scene.render.engine = 'CYCLES'
    # bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    # This has been reduced from 128 since it gives almost identical images with our assets
    scene.cycles.samples = 32

    scene.cycles.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'

    scene.render.layers["RenderLayer"].use_pass_vector = True
    scene.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_material_index = True

    # OSL is NOT supported only in CPU
    # scene.cycles.device = 'GPU'
    # cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
    # cycles_prefs.compute_device_type = "CUDA"


def set_blender_camera(scene, cam, K, W, H):
    """Setup camera intrinsics and extrinsics"""
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    # Set K
    set_blender_camera_from_intrinsics(cam, K)

    # Set camera extrinsic to Identity
    set_blender_camera_extrinsic(cam, Matrix.Identity(3), Vector.Fill(3, 0.0))


def main():
    """Main Function"""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

    smpl_data_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', 'data', 'smpl_data'))
    assert osp.exists(smpl_data_dir), "smpl_data directory {} does not exists".format(smpl_data_dir)

    parser = argparse.ArgumentParser(description='Render SMPL from a single ImageInfo')
    parser.add_argument('image_info_files', type=str, nargs='+', help='path to image info file')
    parser.add_argument('--save_blend', dest='save_blend', action='store_true')
    parser.add_argument('--seed', type=int, help='optional seed value')
    parser.set_defaults(save_blend=False)

    args = parser.parse_args(argv)

    if args.seed is not None:
        print("Using prespecified seed = {}".format(args.seed))
        np.random.seed(args.seed)
    else:
        print("Not using any seed")

    # Setup Scene
    scene = bpy.data.scenes['Scene']
    setup_scene(scene)
    cam = bpy.data.objects['Camera']

    print("---------------------------")
    for mtl in bpy.data.materials:
        print(mtl.name, mtl.type)
    print("---------------------------")

    # Load smpla fbx files
    print("Loading smpl fbx files")
    smpl_obs = load_smpl_fbx_files(smpl_data_dir)

    print("Loading smpl data")
    smpl_data = np.load(osp.join(smpl_data_dir, 'smpl_data.npz'))

    for image_info_file in args.image_info_files:
        assert osp.exists(image_info_file), "File '{}' does not exist".format(image_info_file)
        with open(image_info_file, 'r') as f:
            image_info = json.load(f)

        # image name
        image_name = osp.splitext(osp.basename(image_info['image_file']))[0]
        print("Working on image_name=", image_name)

        # set camera image size, intrinsics and extrinsics
        W = image_info['image_size'][0]
        H = image_info['image_size'][1]
        K_original = Matrix(image_info['image_intrinsic'])
        set_blender_camera(scene, cam, K_original, W, H)
        K = get_camera_intrinsic_from_blender(cam)
        Kinv = K.inverted()

        print("K_original=\n", K_original)
        print("K=\n", K)
        print("Kinv=\n", Kinv)
        assert K == K_original

        # Create a new osl file for each render (Need to verify if it works with multiple processes)
        sh_script = osp.realpath(image_name + '.osl')
        copyfile(osp.join(smpl_data_dir, 'spher_harm.osl'), sh_script)

        print("Adding objects")
        # Loop over all object_infos
        smpl_bodies = []
        for obj_info in image_info['object_infos']:
            if not obj_info['category'].startswith('person_'):
                continue

            gender = obj_info['category'].split('_')[1]
            assert gender in ['male', 'female'], "bad gender {}".format(gender)

            obj_id = obj_info['id']

            cloth_img_path = osp.join(smpl_data_dir, obj_info['shape_file'])
            assert osp.exists(cloth_img_path), "{} does not exist".format(cloth_img_path)

            # set up the material for the object
            material = bpy.data.materials.new(name='Material{:02d}'.format(obj_id))
            material.use_nodes = True
            create_shader_material(material.node_tree, sh_script, cloth_img_path)

            smpl_body = SMPLBody(scene, smpl_obs[gender], material, obj_id)

            shape_param = np.array(obj_info['shape_param'])
            assert shape_param.shape == (10,)

            pose_param = np.concatenate(([0.0, np.pi / 2, 0.0], obj_info['pose_param']))
            assert pose_param.shape == (72,)

            # reset_joint_positions with shape
            scene.objects.active = smpl_body.arm_ob
            smpl_body.reset_joint_positions(shape_param, scene, smpl_data['regression_verts'], smpl_data['joint_regressor'])

            # Put Pelvis at origin
            smpl_body.bone('root').location = - smpl_body.bone('Pelvis').head

            smpl_body.apply_pose_shape(pose_param, shape_param)
            scene.update()

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

            smpl_body.arm_ob.matrix_world = Matrix.Translation(t) * R.to_4x4() * smpl_body.arm_ob.matrix_world

            # spherical harmonics material needs a script to be loaded and compiled
            spherical_harmonics = []
            for _, material in smpl_body.materials.items():
                spherical_harmonics.append(material.node_tree.nodes['Script'])
                spherical_harmonics[-1].filepath = sh_script
                spherical_harmonics[-1].update()

            for _, material in smpl_body.materials.items():
                material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

            # set up random light
            # np.random.seed(0)
            shading_params = .7 * (2 * np.random.rand(9) - 1)
            # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
            shading_params[0] = .5 + .9 * np.random.rand()
            shading_params[1] = -.7 * np.random.rand()

            for ish, coeff in enumerate(shading_params):
                for sc in spherical_harmonics:
                    sc.inputs[ish + 1].default_value = coeff

            smpl_bodies.append(smpl_body)

        if args.save_blend:
            for smpl_ob in smpl_obs.values():
                set_blender_object_hide(smpl_ob, True)
            # Save scene as blend file
            bpy.ops.wm.save_as_mainfile(filepath=image_name + '.blend')
            for smpl_ob in smpl_obs.values():
                set_blender_object_hide(smpl_ob, False)

        # Render
        scene.render.filepath = image_name + '.png'
        bpy.ops.render.render(write_still=True)

        # Now we should delete the smpl bodies
        for smpl_body in smpl_bodies:
            smpl_body.delete_from_scene()

        for material in bpy.data.materials:
            bpy.data.materials.remove(material)


if __name__ == '__main__':
    main()
