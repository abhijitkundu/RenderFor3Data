import math

import bpy
from mathutils import Matrix, Vector
from RenderFor3Data.blender_helper import (get_camera_extrinsic_from_blender,
                                           get_camera_intrinsic_from_blender,
                                           is_rotation_matrix,
                                           rotation_from_two_vectors,
                                           rotation_from_viewpoint)


def print_object_attributes(obj):
    """Helper to print object attributes"""
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))


def main():
    """main function"""

    # Setup Scene
    # scene = bpy.data.scenes['Scene']
    cam = bpy.data.objects['Camera']
    print("K=", get_camera_intrinsic_from_blender(cam))

    t, R = cam.matrix_world.decompose()[0:2]
    R = R.to_matrix()
    print("R=", R)
    print("t=", t)

    assert is_rotation_matrix(R)

    aRb = rotation_from_two_vectors(Vector((0., 0., 1.)), Vector((0., 1., 1.)))
    print("aRb=", aRb)
    bRa = rotation_from_two_vectors(Vector((0., 1., 1.)), Vector((0., 0., 1.)))
    print("bRa=", bRa)

    Rvp = rotation_from_viewpoint(math.pi / 180. * Vector((30., 0., 0.)))
    print("Rvp=", Rvp)


if __name__ == '__main__':
    main()
