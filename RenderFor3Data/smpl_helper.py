'''
SMPL Rendering helper
'''

import bpy
import os.path as osp
from mathutils import Matrix, Vector, Quaternion, Euler
from .blender_helper import deselect_all_objects


class SMPLBody(object):
    """
    SMPL Body
    """
    arm_ob = None
    mesh_ob = None

    def __init__(self, fbx_file, material, obj_idx=0, vertex_segm=None):
        """
        Initialize SMPL shape from fbx and material
        """
        assert osp.exists(fbx_file), "{} does not exist".format(fbx_file)
        bpy.ops.import_scene.fbx(filepath=fbx_file, axis_forward='Y', axis_up='Z', global_scale=100)

        self.arm_ob = bpy.context.active_object
        self.mesh_ob = self.arm_ob.children[0]

        assert '_avg' in self.mesh_ob.name

        self.arm_ob.name = 'Person{:02d}'.format(obj_idx)
        self.mesh_ob.name = 'Person{:02d}_mesh'.format(obj_idx)

        # Disable autosmooth as it can create artifacts
        self.mesh_ob.data.use_auto_smooth = False

        # clear existing animation data
        self.mesh_ob.data.shape_keys.animation_data_clear()
        self.arm_ob.animation_data_clear()

        # create material segmentation
        deselect_all_objects()
        self.mesh_ob.select = True
        bpy.context.scene.objects.active = self.mesh_ob

        # create material segmentation
        if vertex_segm:
            # 0-24 segm labels
            self.materials = create_body_segmentation(vertex_segm, self.mesh_ob, material)
            prob_dressed = {'leftLeg': .5, 'leftArm': .9, 'leftHandIndex1': .01,
                            'rightShoulder': .8, 'rightHand': .01, 'neck': .01,
                            'rightToeBase': .9, 'leftShoulder': .8, 'leftToeBase': .9,
                            'rightForeArm': .5, 'leftHand': .01, 'spine': .9,
                            'leftFoot': .9, 'leftUpLeg': .9, 'rightUpLeg': .9,
                            'rightFoot': .9, 'head': .01, 'leftForeArm': .5,
                            'rightArm': .5, 'spine1': .9, 'hips': .9,
                            'rightHandIndex1': .01, 'spine2': .9, 'rightLeg': .5}
        else:
            # 0 - 1 bg / fg segm labels
            self.materials = {'FullBody': material}
            prob_dressed = {'FullBody': .6}

        # unblocking both the pose and the blendshape limits
        for k in self.mesh_ob.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
            bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

        # TODO Check if we need to do this
        # mesh_ob.active_material = material


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


def load_body_data(smpl_data, ob, gender='female', idx=0):
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

    # compute the number of shape blendshapes in the model
    n_sh_bshapes = len([k for k in ob.data.shape_keys.key_blocks.keys()
                        if k.startswith('Shape')])

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return(cmu_parms, fshapes, name)
