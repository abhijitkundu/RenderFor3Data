import os.path as osp
from pickle import load
from shutil import copyfile
import numpy as np
from random import choice, randint
import bpy
from mathutils import Matrix, Vector
from RenderFor3Data.smpl_helper import SMPLBody, create_shader_material, load_smpl_fbx_files, load_body_data
from RenderFor3Data.blender_helper import (deselect_all_objects,
                                           set_blender_object_hide,
                                           get_camera_intrinsic_from_blender,
                                           set_blender_camera_extrinsic,
                                           set_blender_camera_from_intrinsics,
                                           rotation_from_viewpoint,
                                           rotation_from_two_vectors,
                                           print_blender_object_atrributes)


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
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

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


def set_blender_camera(scene, cam):
    """Setup camera intrinsics"""
    # W = 1600
    # H = 800
    # K = Matrix(((1750.0, 0.0, 800.0),
    #             (0.0, 1750.0, 400.0),
    #             (0.0,    0.0,   1.0)))

    W = 800
    H = 400
    K = Matrix(((875.0, 0.0, 400.0),
                (0.0, 875.0, 200.0),
                (0.0,    0.0,   1.0)))
    # set render size
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    # Set K
    set_blender_camera_from_intrinsics(cam, K)

    # Set camera extrinsic to Identity
    set_blender_camera_extrinsic(cam, Matrix.Identity(3), Vector.Fill(3, 0.0))

    return W, H, K


def main():
    """Main Function"""
    smpl_data_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', 'data', 'smpl_data'))
    assert osp.exists(smpl_data_dir), "smpl_data directory {} does not exists".format(smpl_data_dir)

    image_name = 'temp'

    genders = ['female', 'male']

    # Load all gender specific stuff
    texture_paths = {}
    shape_param_dist = {}
    for gender in genders:
        # grab clothing names
        clothing_split = 'train'  # choices = ['train', test', 'all']
        clothing_option = 'all'  # choices = ['nongrey', grey', 'all']

        with open(osp.join(smpl_data_dir, 'textures', '%s_%s.txt' % (gender, clothing_split))) as f:
            txt_paths = f.read().splitlines()

        if clothing_option == 'nongrey':
            txt_paths = [k for k in txt_paths if 'nongrey' in k]
        elif clothing_option == 'grey':
            txt_paths = [k for k in txt_paths if 'nongrey' not in k]

        assert txt_paths, "Need to have atleast 1 texture"
        print("Using {} textures for {}".format(len(txt_paths), gender))
        texture_paths[gender] = txt_paths

        beta_stds = np.load(osp.join(smpl_data_dir, ('%s_beta_stds.npy' % gender)))
        shape_param_dist[gender] = beta_stds

    print("Loading smpl data")
    smpl_data = np.load(osp.join(smpl_data_dir, 'smpl_data.npz'))
    print("# of smpl_data.files = {}".format(len(smpl_data.files)))

    print("Loading segm_per_v_overlap.pkl")
    with open(osp.join(smpl_data_dir, 'segm_per_v_overlap.pkl'), 'rb') as f:
        vsegm = load(f)

    # Create a new osl file for each render (Need to verify if it works with multiple processes)
    sh_script = osp.realpath(image_name + '.osl')
    copyfile(osp.join(smpl_data_dir, 'spher_harm.osl'), sh_script)

    # Setup Scene
    scene = bpy.data.scenes['Scene']
    setup_scene(scene)

    cam = bpy.data.objects['Camera']
    W, H, _ = set_blender_camera(scene, cam)
    K = get_camera_intrinsic_from_blender(cam)
    Kinv = K.inverted()
    print("K=\n", K)

    num_of_mocap_seqs = sum(1 for seq in smpl_data.files if seq.startswith('pose_'))
    print("num_of_mocap_seqs=", num_of_mocap_seqs)

    # Load smpla fbx files
    smpl_obs = load_smpl_fbx_files(smpl_data_dir)

    num_of_persons = 10
    for obj_id in range(num_of_persons):
        gender = choice(genders)
        print(shape_param_dist[gender].shape)

        mocap_seqid = randint(0, num_of_mocap_seqs - 1)
        print("loading body data from seq {}".format(mocap_seqid))
        cmu_params, fshapes, _ = load_body_data(smpl_data, idx=mocap_seqid, gender=gender, num_of_shape_params=10)

        assert 'poses' in cmu_params
        assert 'trans' in cmu_params
        assert cmu_params['poses'].shape[1] == 72
        assert cmu_params['trans'].shape[1] == 3

        print(cmu_params['poses'].shape)
        print(cmu_params['trans'].shape)
        print(fshapes.shape)

        # set up the material for the object
        material = bpy.data.materials.new(name='Material{:02d}'.format(obj_id))
        material.use_nodes = True

        cloth_img_path = osp.join(smpl_data_dir, choice(texture_paths[gender]))
        assert osp.exists(cloth_img_path), "{} does not exist".format(cloth_img_path)
        create_shader_material(material.node_tree, sh_script, cloth_img_path)

        smpl_body = SMPLBody(scene, smpl_obs[gender], material, obj_id, vertex_segm=vsegm)

        # TODO Check if this is required
        # mesh_ob.active_material = material

        # Get random shape
        shape = choice(fshapes)

        # reset_joint_positions with shape
        scene.objects.active = smpl_body.arm_ob
        smpl_body.reset_joint_positions(shape, scene, smpl_data['regression_verts'], smpl_data['joint_regressor'])

        # Put Pelvis at origin
        smpl_body.bone('root').location = - smpl_body.bone('Pelvis').head

        pose = choice(cmu_params['poses'])
        pose[0] = 0
        pose[1] = np.pi / 2
        pose[2] = 0
        smpl_body.apply_pose_shape(pose, shape)
        scene.update()

        # rotation by viewwpoint
        center_proj_ray = Kinv * Vector((np.random.uniform(5.0, W - 5.0), 
                                         np.random.uniform(5.0, H - 5.0),
                                        1.0))
        # Rotation to account for object not along principal axes
        Rdelta = rotation_from_two_vectors(Vector((0., 0., 1.)), center_proj_ray)

        Rvp = rotation_from_viewpoint((np.random.uniform(-np.pi, np.pi), 
                                       np.random.uniform(-np.pi / 6, np.pi / 6), 
                                       np.random.uniform(-np.pi / 12, np.pi / 12)))
       
        # Compute final object pose as R|t = R_delta * [R_vp| t_vp]
        R = Rdelta * Rvp
        t = Rdelta * Vector((0., 0., np.random.uniform(3.0, 30.0)))

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
        shading_params = .7 * (2 * np.random.rand(9) - 1)
        # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
        shading_params[0] = .5 + .9 * np.random.rand()
        shading_params[1] = -.7 * np.random.rand()

        for ish, coeff in enumerate(shading_params):
            for sc in spherical_harmonics:
                sc.inputs[ish + 1].default_value = coeff

    # Hide the template smpl objects
    for smpl_ob in smpl_obs.values():
        set_blender_object_hide(smpl_ob)

    # Save scene as blend file
    bpy.ops.wm.save_as_mainfile(filepath=image_name + '.blend')

    # Render
    scene.render.filepath = image_name + '.png'
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
