'''
SMPL Rendering helper
'''

import os.path as osp
import numpy as np

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from .blender_helper import deselect_all_objects

class SMPLBody(object):
    """
    SMPL Body
    """
    arm_ob = None
    mesh_ob = None
    bone_prefix = None

    # order
    part_match = {'root': 'root', 'bone_00': 'Pelvis', 'bone_01': 'L_Hip', 'bone_02': 'R_Hip',
                  'bone_03': 'Spine1', 'bone_04': 'L_Knee', 'bone_05': 'R_Knee', 'bone_06': 'Spine2',
                  'bone_07': 'L_Ankle', 'bone_08': 'R_Ankle', 'bone_09': 'Spine3', 'bone_10': 'L_Foot',
                  'bone_11': 'R_Foot', 'bone_12': 'Neck', 'bone_13': 'L_Collar', 'bone_14': 'R_Collar',
                  'bone_15': 'Head', 'bone_16': 'L_Shoulder', 'bone_17': 'R_Shoulder', 'bone_18': 'L_Elbow',
                  'bone_19': 'R_Elbow', 'bone_20': 'L_Wrist', 'bone_21': 'R_Wrist', 'bone_22': 'L_Hand', 'bone_23': 'R_Hand'}

    def __init__(self, scene, ref_smpl_ob, material, obj_idx=0, vertex_segm=None):
        """
        Initialize SMPL shape from ref smpl ob and material
        """

        assert ref_smpl_ob.type == 'ARMATURE'
        assert len(ref_smpl_ob.children) == 1
        assert ref_smpl_ob.children[0].type == 'MESH'

        deselect_all_objects()
        # Somehow the they need to unhidden
        assert not ref_smpl_ob.hide
        assert not ref_smpl_ob.children[0].hide

        ref_smpl_ob.select = True
        ref_smpl_ob.children[0].select = True

        bpy.ops.object.duplicate()

        for ob in bpy.context.selected_objects:
            if ob.type == 'ARMATURE':
                self.arm_ob = ob
            elif ob.type == 'MESH':
                self.mesh_ob = ob
            else:
                raise RuntimeError('Bad object type {}'.format(ob.type))


        self.arm_ob.name = 'Person{:02d}'.format(obj_idx)
        self.mesh_ob.name = 'Person{:02d}_mesh'.format(obj_idx)

        # UnHide these template objects
        self.arm_ob.hide = False
        self.arm_ob.hide_render = False

        self.mesh_ob.hide = False
        self.mesh_ob.hide_render = False

        # make sure to Disable autosmooth as it can create artifacts
        assert not self.mesh_ob.data.use_auto_smooth

        self.bone_prefix = common_prefix(self.arm_ob.pose.bones.keys())
        assert self.bone_prefix == 'm_avg_' or self.bone_prefix == 'f_avg_'

        # create material segmentation
        deselect_all_objects()
        self.mesh_ob.select = True
        scene.objects.active = self.mesh_ob

        # create material segmentation
        if vertex_segm:
            # 0-24 segm labels
            self.materials = create_body_segmentation(vertex_segm, self.mesh_ob, material)
            # prob_dressed = {'leftLeg': .5, 'leftArm': .9, 'leftHandIndex1': .01,
            #                 'rightShoulder': .8, 'rightHand': .01, 'neck': .01,
            #                 'rightToeBase': .9, 'leftShoulder': .8, 'leftToeBase': .9,
            #                 'rightForeArm': .5, 'leftHand': .01, 'spine': .9,
            #                 'leftFoot': .9, 'leftUpLeg': .9, 'rightUpLeg': .9,
            #                 'rightFoot': .9, 'head': .01, 'leftForeArm': .5,
            #                 'rightArm': .5, 'spine1': .9, 'hips': .9,
            #                 'rightHandIndex1': .01, 'spine2': .9, 'rightLeg': .5}
        else:
            # 0 - 1 bg / fg segm labels
            bpy.ops.object.material_slot_remove()
            self.materials = {'FullBody': material}
            bpy.ops.object.material_slot_add()
            self.mesh_ob.material_slots[-1].material = self.materials['FullBody']
            # prob_dressed = {'FullBody': .6}

    def bone(self, bone_type='Pelvis'):
        """returns a particluar bone"""
        if isinstance(bone_type, str):
            return self.arm_ob.pose.bones[self.bone_prefix + bone_type]
        elif isinstance(bone_type, int):
            return self.arm_ob.pose.bones[self.bone_prefix + self.part_match['bone_%02d' % bone_type]]
        else:
            raise TypeError

    def num_of_shape_params(self):
        """num of blend shapes"""
        return len([k for k in self.mesh_ob.data.shape_keys.key_blocks.keys() if k.startswith('Shape')])

    def apply_pose_shape(self, pose, shape, frame=None):
        """
        apply trans pose and shape to character
        transform pose into rotation matrices (for pose) and pose blendshapes
        """
        mrots, bsh = rodrigues2bshapes(pose)

        # set the pose of each bone to the quaternion specified by pose
        for ibone, mrot in enumerate(mrots):
            bone = self.bone(ibone)
            bone.rotation_quaternion = Matrix(mrot).to_quaternion()
            if frame is not None:
                bone.keyframe_insert('rotation_quaternion', frame=frame)
                bone.keyframe_insert('location', frame=frame)

        # apply pose blendshapes
        for ibshape, bshape in enumerate(bsh):
            self.mesh_ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
            if frame is not None:
                self.mesh_ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

        # apply shape blendshapes
        for ibshape, shape_elem in enumerate(shape):
            self.mesh_ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
            if frame is not None:
                self.mesh_ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

    def reset_joint_positions(self, shape, scene, reg_ivs, joint_reg):
        """reset the joint positions of the character according to its new shape"""
        # since the regression is sparse, only the relevant vertex
        #     elements (joint_reg) and their indices (reg_ivs) are loaded
        reg_vs = np.empty((len(reg_ivs), 3))  # empty array to hold vertices to regress from
        # zero the pose and trans to obtain joint positions in zero pose
        self.apply_pose_shape(np.zeros(72), shape)

        # obtain a mesh after applying modifiers
        # bpy.ops.wm.memory_statistics()
        # me holds the vertices after applying the shape blendshapes
        me = self.mesh_ob.to_mesh(scene, True, 'PREVIEW')

        # fill the regressor vertices matrix
        for iiv, iv in enumerate(reg_ivs):
            reg_vs[iiv] = me.vertices[iv].co
        bpy.data.meshes.remove(me)

        # regress joint positions in rest pose
        joint_xyz = joint_reg.dot(reg_vs)
        # adapt joint positions in rest pose
        self.arm_ob.hide = False
        bpy.ops.object.mode_set(mode='EDIT')
        self.arm_ob.hide = True
        for ibone in range(24):
            bb = self.arm_ob.data.edit_bones[self.bone_prefix + self.part_match['bone_%02d' % ibone]]
            bboffset = bb.tail - bb.head
            bb.head = joint_xyz[ibone]
            bb.tail = bb.head + bboffset
        bpy.ops.object.mode_set(mode='OBJECT')


def Rodrigues(rotvec):
    """computes rotation matrix through Rodrigues formula as in cv2.Rodrigues"""
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


def rodrigues2bshapes(pose):
    """transformation between pose and blendshapes"""
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return(mat_rots, bshapes)

def load_smpl_fbx_files(smpl_data_dir):
    """load smpl fbx files"""
    smpl_obs = {}
    for gender in ['female', 'male']:
        fbx_file = osp.join(smpl_data_dir, 'basicModel_{}_lbs_10_207_0_v1.0.2.fbx'.format(gender[0]))
        assert osp.exists(fbx_file), "{} does not exist".format(fbx_file)
        bpy.ops.import_scene.fbx(filepath=fbx_file, axis_forward='Y', axis_up='Z', global_scale=100)

        arm_ob = bpy.context.active_object
        mesh_ob = arm_ob.children[0]

        mesh_prefix = mesh_ob.name + '_'
        assert mesh_prefix == '{}_avg_'.format(gender[0])

        arm_ob.name = gender
        mesh_ob.name = '{}_mesh'.format(gender)
        mesh_ob.data.shape_keys.name = '{}_mesh'.format(gender)

        # Disable autosmooth as it can create artifacts
        mesh_ob.data.use_auto_smooth = False

        # clear existing animation data
        mesh_ob.data.shape_keys.animation_data_clear()
        arm_ob.animation_data_clear()

        # Hide these template objects from render
        arm_ob.hide_render = True
        mesh_ob.hide_render = True

        smpl_obs[gender] = arm_ob

        # unblocking both the pose and the blendshape limits
        for k in mesh_ob.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys[mesh_ob.data.shape_keys.name].key_blocks[k].slider_min = -10
            bpy.data.shape_keys[mesh_ob.data.shape_keys.name].key_blocks[k].slider_max = 10

    return smpl_obs


def common_prefix(strings):
    """ Find the longest string that is a prefix of all the strings."""
    if not strings:
        return ''
    prefix = strings[0]
    for s in strings:
        if len(s) < len(prefix):
            prefix = prefix[:len(s)]
        if not prefix:
            return ''
        for i in range(len(prefix)):
            if prefix[i] != s[i]:
                prefix = prefix[:i]
                break
    return prefix

def create_body_segmentation(vertex_segm, ob, material):
    """
    material is the input material map for different objects
    """

    # smpl body part information
    sorted_parts = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
                    'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
                    'neck', 'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm',
                    'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']

    part2num = {part: (ipart + 1) for ipart, part in enumerate(sorted_parts)}

    materials = {}
    vgroups = {}
    bpy.ops.object.material_slot_remove()
    parts = sorted(vertex_segm.keys())
    for part in parts:
        vs = vertex_segm[part]
        vgroups[part] = ob.vertex_groups.new(part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)
        materials[part] = material.copy()
        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        ob.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')

    return materials


def create_shader_material(tree, sh_path, cloth_image_path):
    """
    creation of the spherical harmonics material, using an OSL script
    """
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    uv_xform.operation = 'AVERAGE'

    cloth_img = bpy.data.images.load(cloth_image_path)
    assert cloth_img
    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    uv_im.image = cloth_img

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


def load_body_data(smpl_data, idx=0, gender='female', num_of_shape_params=10):
    """
    load MoSHed data from CMU Mocap (only the given idx is loaded)
    """
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

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :num_of_shape_params]

    return(cmu_parms[name], fshapes, name)
