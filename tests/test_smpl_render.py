import os.path as osp
import bpy
import numpy as np
from mathutils import Matrix, Vector
from RenderFor3Data.blender_helper import print_blender_object_atrributes
from RenderFor3Data.smpl_helper import (rodrigues, rodrigues2bshapes)

def verts_as_numpy_arrray(vertices):
    mverts_co = np.zeros((len(vertices) * 3), dtype=np.float)
    vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(vertices), 3))

def main():
    smpl_model_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', 'data', 'SMPLmodels'))

    # Setup Scene
    scene = bpy.data.scenes['Scene']

       # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # clear default Cube object
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    fbx_file = osp.join(smpl_model_dir, 'models', 'male_lbs_10_207_0_v1.0.2.fbx')
    assert osp.exists(fbx_file), "{} does not exist".format(fbx_file)
    bpy.ops.import_scene.fbx(filepath=fbx_file, use_manual_orientation=True, axis_forward='Y', axis_up='Z', global_scale=100)

    arm_ob = bpy.context.active_object
    mesh_ob = arm_ob.children[0]

    print(arm_ob.matrix_world)
    print(mesh_ob.matrix_world)

    verts = verts_as_numpy_arrray(mesh_ob.data.vertices)
    print("verts=\n", verts)
    print("verts.shape=", verts.shape)
    print("min=", np.amin(verts, axis=0))
    print("max=", np.amax(verts, axis=0))

    mat_rots, bshapes = rodrigues2bshapes(np.zeros(72))
    # print(mat_rots.shape)
    print(mat_rots)
    print(bshapes.shape)
    print(bshapes)

    print_blender_object_atrributes(mesh_ob.data.shape_keys.key_blocks)
    for key in sorted(mesh_ob.data.shape_keys.key_blocks.keys()):
        print(key)

    key_block = mesh_ob.data.shape_keys.key_blocks['Shape000']
    print_blender_object_atrributes(key_block)

    key_block_data = verts_as_numpy_arrray(key_block.data)
    print("key_block_data=\n", key_block_data)
    print("key_block_data.shape=", key_block_data.shape)
    print("min=", np.amin(key_block_data, axis=0))
    print("max=", np.amax(key_block_data, axis=0))

    # Save scene as blend file
    bpy.ops.wm.save_as_mainfile(filepath='test.blend')

if __name__ == '__main__':
    main()
