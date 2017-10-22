"""
This script tests blender camera projection model with MVG camera projection model
"""

import bpy
from mathutils import Matrix, Vector

from RenderFor3Data.blender_helper import (
    get_camera_intrinsic_from_blender,
    get_camera_extrinsic_from_blender,
    project_by_object_utils,
    set_blender_camera_from_intrinsics,
    print_blender_object_atrributes
)

def main():
    """main function"""
    print("Setting up scene")

    # Setup Scene
    scene = bpy.data.scenes['Scene']
    # scene.render.engine = 'CYCLES'
    # bpy.data.materials['Material'].use_nodes = True
    # scene.cycles.shading_system = True
    # scene.use_nodes = True

    cam = bpy.data.objects['Camera']

    print_blender_object_atrributes(cam.data)

    K = get_camera_intrinsic_from_blender(cam)
    RT = get_camera_extrinsic_from_blender(cam)
    P = K * RT

    print("K")
    print(K)
    print("RT")
    print(RT)
    print("P")
    print(P)

    print("===============================================")
    e1 = Vector((1, 0,    0, 1))
    e2 = Vector((0, 1,    0, 1))
    e3 = Vector((0, 0,    1, 1))
    O = Vector((0, 0,    0, 1))

    p1 = P * e1
    p1 /= p1[2]
    print("Projected by KRT :", p1)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e1[0:3])))

    print("----------------------------")

    p2 = P * e2
    p2 /= p2[2]
    print("Projected by KRT :", p2)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e2[0:3])))

    print("----------------------------")

    p3 = P * e3
    p3 /= p3[2]
    print("Projected by KRT :", p3)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e3[0:3])))

    print("----------------------------")

    pO = P * O
    pO /= pO[2]
    print("Projected by KRT :", pO)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(O[0:3])))


    print("===============================================")
    print("===============================================")
    print("===============================================")

    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100

    K = Matrix(((1050.0,   0, 480.0),
                (0,   1050.0, 270.0),
                (0,        0,     1)))
    print("K")
    print(K)

    set_blender_camera_from_intrinsics(cam, K)

    print("K")
    print(get_camera_intrinsic_from_blender(cam))

    P = K * RT
    p1 = P * e1
    p1 /= p1[2]
    print("Projected by KRT :", p1)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e1[0:3])))

    print("----------------------------")

    p2 = P * e2
    p2 /= p2[2]
    print("Projected by KRT :", p2)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e2[0:3])))

    print("----------------------------")

    p3 = P * e3
    p3 /= p3[2]
    print("Projected by KRT :", p3)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e3[0:3])))

    print("----------------------------")

    pO = P * O
    pO /= pO[2]
    print("Projected by KRT :", pO)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(O[0:3])))

    print("===============================================")
    print("===============================================")
    print("===============================================")

    scene.render.resolution_x = 1242
    scene.render.resolution_y = 375
    scene.render.resolution_percentage = 100

    K = Matrix(((721.5377, 0.0, 609.5593),
                (0.0, 721.5377, 172.854),
                (0.0,      0.0,     1.0)))
    print("K")
    print(K)

    set_blender_camera_from_intrinsics(cam, K)

    print("K")
    print(get_camera_intrinsic_from_blender(cam))

    P = K * RT
    p1 = P * e1
    p1 /= p1[2]
    print("Projected by KRT :", p1)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e1[0:3])))

    print("----------------------------")

    p2 = P * e2
    p2 /= p2[2]
    print("Projected by KRT :", p2)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e2[0:3])))

    print("----------------------------")

    p3 = P * e3
    p3 /= p3[2]
    print("Projected by KRT :", p3)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(e3[0:3])))

    print("----------------------------")

    pO = P * O
    pO /= pO[2]
    print("Projected by KRT :", pO)
    print("proj by bpy_utils:", project_by_object_utils(cam, Vector(O[0:3])))


if __name__ == '__main__':
    main()
