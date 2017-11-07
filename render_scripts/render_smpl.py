import sys
import os
import random
import math
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from glob import glob
from random import choice
from pickle import load
from shutil import copyfile
import bpy
from mathutils import Matrix, Vector, Quaternion, Euler
from bpy_extras.object_utils import world_to_camera_view as world2cam

def setState0():
    for ob in bpy.data.objects.values():
        ob.select = False
    bpy.context.scene.objects.active = None


sorted_parts = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
                'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
                'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm',
                'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']
# order
part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
              'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
              'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
              'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
              'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
              'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand', 'bone_23': 'R_Hand'}

part2num = {part: (ipart + 1) for ipart, part in enumerate(sorted_parts)}

def create_segmentation(ob, segm_overlap_file):
    """
    create one material per part as defined in a pickle with the segmentation
    this is useful to render the segmentation in a material pass
    """
    materials = {}
    vgroups = {}
    with open(segm_overlap_file, 'rb') as f:
        vsegm = load(f)
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = bpy.data.materials['Material'].copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return(materials)

def create_sh_material(tree, sh_path, img=None):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    script = tree.nodes.new('ShaderNodeScript')
    script.location = -230, 400
    script.mode = 'EXTERNAL'
    script.filepath = sh_path  # 'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])

def init_scene(scene, fbx_file):
    # load fbx model
    bpy.ops.import_scene.fbx(filepath=fbx_file, axis_forward='Y', axis_up='Z', global_scale=100)

    arm_ob = bpy.context.active_object
    obname = arm_ob.children[0].name
    assert '_avg' in obname
    ob = bpy.data.objects[obname]
    ob.data.use_auto_smooth = False  # autosmooth creates artifacts

    # assign the existing spherical harmonics material
    ob.active_material = bpy.data.materials['Material']

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    scn = bpy.context.scene

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_material_index = True

    # clear existing animation data
    ob.data.shape_keys.animation_data_clear()
    arm_ob = bpy.data.objects['Armature']
    arm_ob.animation_data_clear()

    return(ob, obname, arm_ob)

def load_body_data(smpl_data, ob, obname, gender='female', idx=0):
    # load MoSHed data from CMU Mocap (only the given idx is loaded)

    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {'poses': smpl_data[seq],
                                                   'trans': smpl_data[seq.replace('pose_', 'trans_')]}

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms, fshapes, name)


import time
start_time = None


def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))


def main():
    # time logging
    global start_time
    start_time = time.time()

    import argparse

    # parse commandline arguments
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    log_message(argv)

    parser = argparse.ArgumentParser(description='Generate synth SMPL images.')
    parser.add_argument('-s', '--split', choices=['train', 'test', 'all'], default='train', help='Split to use')
    args = parser.parse_args(argv)

    smpl_data_folder = '/home/abhijit/Scratchspace/SURREAL/smpl_data'
    smpl_data_filename = 'smpl_data.npz'
    clothing_option = 'all'  # grey, nongrey or all
    split = 'train'

    genders = {0: 'female', 1: 'male'}
    # pick random gender
    gender = choice(genders)
    log_message("Gender = %s" % gender)

    # grab clothing names
    with open(join(smpl_data_folder, 'textures', '%s_%s.txt' % (gender, args.split))) as f:
        txt_paths = f.read().splitlines()
    log_message("Found %d textures" % len(txt_paths))

    # if using only one source of clothing
    log_message("clothing: %s" % clothing_option)
    if clothing_option == 'nongrey':
        txt_paths = [k for k in txt_paths if 'nongrey' in k]
    elif clothing_option == 'grey':
        txt_paths = [k for k in txt_paths if 'nongrey' not in k]
    log_message("Final # of textures =  %d" % len(txt_paths))

    log_message("Loading parts segmentation")
    beta_stds = np.load(join(smpl_data_folder, ('%s_beta_stds.npy' % gender)))
    log_message("beta_stds.shape = {}".format(beta_stds.shape))

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))
    log_message("# of smpl_data.files = {}".format(len(smpl_data.files)))

    log_message("Setup Blender")

    # create copy-spher.harm. directory if not exists
    sh_dst = realpath('sh.osl')
    copyfile(join(smpl_data_folder, 'spher_harm.osl'), sh_dst)

    # Setup Scene
    scene = bpy.data.scenes['Scene']
    scene.render.engine = 'CYCLES'
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    # random clothing texture
    cloth_img_name = choice(txt_paths)
    cloth_img_name = join(smpl_data_folder, cloth_img_name)
    cloth_img = bpy.data.images.load(cloth_img_name)

    log_message("Building materials tree")
    mat_tree = bpy.data.materials['Material'].node_tree
    create_sh_material(mat_tree, sh_dst, cloth_img)

    log_message("Initializing scene")

    fbx_file = join(smpl_data_folder, 'basicModel_%s_lbs_10_207_0_v1.0.2.fbx' % gender[0])
    ob, obname, arm_ob = init_scene(scene, fbx_file)
    cam_ob = bpy.data.objects['Camera']

    setState0()
    ob.select = True
    bpy.context.scene.objects.active = ob
    segmented_materials = True  # True: 0-24, False: expected to have 0-1 bg/fg

    log_message("Creating materials segmentation")
    # create material segmentation
    if segmented_materials:
        materials = create_segmentation(ob, join(smpl_data_folder, 'segm_per_v_overlap.pkl'))
        prob_dressed = {'leftLeg': .5, 'leftArm': .9, 'leftHandIndex1': .01,
                        'rightShoulder': .8, 'rightHand': .01, 'neck': .01,
                        'rightToeBase': .9, 'leftShoulder': .8, 'leftToeBase': .9,
                        'rightForeArm': .5, 'leftHand': .01, 'spine': .9,
                        'leftFoot': .9, 'leftUpLeg': .9, 'rightUpLeg': .9,
                        'rightFoot': .9, 'head': .01, 'leftForeArm': .5,
                        'rightArm': .5, 'spine1': .9, 'hips': .9,
                        'rightHandIndex1': .01, 'spine2': .9, 'rightLeg': .5}
    else:
        materials = {'FullBody': bpy.data.materials['Material']}
        prob_dressed = {'FullBody': .6}

    # unblocking both the pose and the blendshape limits
    for k in ob.data.shape_keys.key_blocks.keys():
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    log_message("Loading body data")
    cmu_parms, fshapes, name = load_body_data(smpl_data, ob, obname, idx=0, gender=gender)


    log_message("Loaded body data for %s" % name)

    nb_fshapes = len(fshapes)
    if args.split == 'train':
        fshapes = fshapes[:int(nb_fshapes * 0.8)]
    elif args.split == 'test':
        fshapes = fshapes[int(nb_fshapes * 0.8):]

    # pick random real body shape
    # shape = choice(fshapes)  # +random_shape(.5) can add noise
    #shape = random_shape(3.) # random body shape

    # example shapes
    #shape = np.zeros(10) #average
    #shape = np.array([ 2.25176191, -3.7883464 ,  0.46747496,  3.89178988,  2.20098416,  0.26102114, -3.07428093,  0.55708514, -3.94442258, -2.88552087]) #fat
    #shape = np.array([-2.26781107,  0.88158132, -0.93788176, -0.23480508,  1.17088298,  1.55550789,  0.44383225,  0.37688275, -0.27983086,  1.77102953]) #thin
    #shape = np.array([ 0.00404852,  0.8084637 ,  0.32332591, -1.33163664,  1.05008727,  1.60955275,  0.22372946, -0.10738459,  0.89456312, -1.22231216]) #short
    #shape = np.array([ 3.63453289,  1.20836171,  3.15674431, -0.78646793, -1.93847355, -0.32129994, -0.97771656,  0.94531640,  0.52825811, -0.99324327]) #tall

    ndofs = 10
    scene.objects.active = arm_ob
    orig_trans = np.asarray(arm_ob.pose.bones[obname + '_Pelvis'].location).copy()

    # spherical harmonics material needs a script to be loaded and compiled
    scs = []
    for _, material in materials.items():
        scs.append(material.node_tree.nodes['Script'])
        scs[-1].filepath = sh_dst
        scs[-1].update()

    arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    for _, material in materials.items():
        material.node_tree.nodes['Vector Math'].inputs[1].default_value[:2] = (0, 0)

    # random light
    sh_coeffs = .7 * (2 * np.random.rand(9) - 1)
    sh_coeffs[0] = .5 + .9 * np.random.rand()  # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[1] = -.7 * np.random.rand()

    for ish, coeff in enumerate(sh_coeffs):
        for sc in scs:
            sc.inputs[ish + 1].default_value = coeff

    bpy.ops.wm.save_as_mainfile(filepath='old.blend')

    # Render
    scene.render.filepath = 'old.png'
    bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    main()
