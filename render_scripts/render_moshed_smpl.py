import os.path as osp
from pickle import load
from shutil import copyfile
import numpy as np
from random import choice
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
from RenderFor3Data.smpl_helper import create_body_segmentation, create_shader_material
from RenderFor3Data.blender_helper import (deselect_all_objects,
                                           get_camera_intrinsic_from_blender,
                                           set_blender_camera_extrinsic,
                                           set_blender_camera_from_intrinsics,
                                           spherical_to_cartesian,
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

    # OSL is supported only in CPU
    scene.cycles.device = 'GPU'
    cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = "CUDA"


def set_blender_camera(scene, cam):
    """Setup camera intrinsics"""
    W = 1600
    H = 800
    K = Matrix(((1750.0, 0.0, 800.0),
                (0.0, 1750.0, 400.0),
                (0.0,    0.0,   1.0)))
    # set render size
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    # Set K
    set_blender_camera_from_intrinsics(cam, K)

    # Set camera extrinsic to Identity
    set_blender_camera_extrinsic(cam, Matrix.Identity(3), Vector.Fill(3, 0.0))


def add_person_to_scene(smpl_data_dir, gender, obj_idx):
    """
    Add a SMPL person to scene
    """
    if gender == 'male':
        fbx_file = osp.join(smpl_data_dir, 'basicModel_m_lbs_10_207_0_v1.0.2.fbx')
    elif gender == 'female':
        fbx_file = osp.join(smpl_data_dir, 'basicModel_f_lbs_10_207_0_v1.0.2.fbx')
    else:
        raise NotImplementedError

    assert osp.exists(fbx_file), "{} does not exist".format(fbx_file)

    bpy.ops.import_scene.fbx(filepath=fbx_file, axis_forward='Y', axis_up='Z', global_scale=100)

    arm_ob = bpy.context.active_object
    mesh_ob = arm_ob.children[0]
    assert '_avg' in mesh_ob.name

    arm_ob.name = 'Person{:02d}'.format(obj_idx)
    mesh_ob.name = 'Person{:02d}_mesh'.format(obj_idx)
    mesh_ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # clear existing animation data
    mesh_ob.data.shape_keys.animation_data_clear()
    arm_ob.animation_data_clear()

    return(mesh_ob, arm_ob)

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

    gender = choice(genders)
    print(shape_param_dist[gender].shape)

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
    set_blender_camera(scene, cam)
    K = get_camera_intrinsic_from_blender(cam)
    print("K=\n", K)

    obj_id = 0

    # set up the material for the object
    material = bpy.data.materials.new(name='Material{:02d}'.format(obj_id))
    material.use_nodes = True

    cloth_img_path = osp.join(smpl_data_dir, choice(texture_paths[gender]))
    assert osp.exists(cloth_img_path), "{} does not exist".format(cloth_img_path)
    create_shader_material(material.node_tree, sh_script, cloth_img_path)

    mesh_ob, arm_ob = add_person_to_scene(smpl_data_dir, 'male', obj_id)
    # assign the existing spherical harmonics material
    # mesh_ob.active_material = bpy.data.materials['Material']

    deselect_all_objects()
    mesh_ob.select = True
    scene.objects.active = mesh_ob
    segmented_materials = True  # True: 0-24, False: expected to have 0-1 bg/fg

    # create material segmentation
    if segmented_materials:
        materials = create_body_segmentation(vsegm, mesh_ob, material)
        prob_dressed = {'leftLeg': .5, 'leftArm': .9, 'leftHandIndex1': .01,
                        'rightShoulder': .8, 'rightHand': .01, 'neck': .01,
                        'rightToeBase': .9, 'leftShoulder': .8, 'leftToeBase': .9,
                        'rightForeArm': .5, 'leftHand': .01, 'spine': .9,
                        'leftFoot': .9, 'leftUpLeg': .9, 'rightUpLeg': .9,
                        'rightFoot': .9, 'head': .01, 'leftForeArm': .5,
                        'rightArm': .5, 'spine1': .9, 'hips': .9,
                        'rightHandIndex1': .01, 'spine2': .9, 'rightLeg': .5}
    else:
        materials = {'FullBody': material}
        prob_dressed = {'FullBody': .6}

    # unblocking both the pose and the blendshape limits
    for k in mesh_ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    mesh_ob.active_material = material

    # spherical harmonics material needs a script to be loaded and compiled
    spherical_harmonics = []
    for _, material in materials.items():
        spherical_harmonics.append(material.node_tree.nodes['Script'])
        spherical_harmonics[-1].filepath = sh_script
        spherical_harmonics[-1].update()

    for _, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

    # set up random light
    shading_params = .7 * (2 * np.random.rand(9) - 1)
    # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    shading_params[0] = .5 + .9 * np.random.rand()
    shading_params[1] = -.7 * np.random.rand()

    for ish, coeff in enumerate(shading_params):
        for sc in spherical_harmonics:
            sc.inputs[ish + 1].default_value = coeff

    # Save scene as blend file
    bpy.ops.wm.save_as_mainfile(filepath=image_name + '.blend')

    # Render
    scene.render.filepath = image_name + '.png'
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
