"""
This script tests blender camera projection model with MVG camera projection model
"""

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def get_camera_intrinsic_from_blender(cam):
    """Get the camera intrinsic matrix K for blender camera"""
    camd = cam.data
    assert (camd.lens_unit != 'FOV')
    f_in_mm = camd.lens
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100

    W = scene.render.resolution_x * scale
    H = scene.render.resolution_y * scale

    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camd.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = W / sensor_width_in_mm / pixel_aspect_ratio
        s_v = H / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = W / sensor_width_in_mm
        s_v = H * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = W / 2 + camd.shift_x * W
    v_0 = H / 2 - camd.shift_y * W
    skew = 0  # only use rectangular pixels

    K = Matrix(((alpha_u, skew, u_0),
                (0,    alpha_v, v_0),
                (0,          0,   1)))
    return K


def get_camera_extrinsic_from_blender(cam):
    """
    Returns camera extrinsic [R|t] (rotation and translation matrices) from Blender.
    There are 3 coordinate systems involved:
    1. The World coordinates: "world"
       - right-handed
    2. The Blender camera coordinates: "bcam"
       - x is horizontal
       - y is up
       - right-handed: negative z look-at direction
    3. The desired computer vision camera coordinates: "cv"
       - x is horizontal
       - y is down (to align to the actual pixel coordinates
         used in digital images)
       - right-handed: positive z look-at direction
    """
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1,  0,  0),
                        (0, -1,  0),
                        (0,  0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv * R_world2bcam
    T_world2cv = R_bcam2cv * T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def project_by_object_utils(cam, point):
    """
    Get pixel coord using world_to_camera_view function in bpy.object_utils library
    """
    scene = bpy.context.scene
    co_2d = world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


def set_blender_camera_from_intrinsics(cam, K):
    """Set blender camera from intrinsic matrix K"""
    camd = cam.data
    assert (camd.lens_unit != 'FOV')
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    W = (scene.render.resolution_x * scale)
    H = (scene.render.resolution_y * scale)

    fx = float(K[0][0])
    fy = float(K[1][1])
    cx = float(K[0][2])
    cy = float(K[1][2])

    maxdim = max(W, H)

    camd.shift_x = (cx - W / 2) / maxdim
    camd.shift_y = ((H - cy) - H / 2) / maxdim  # (TODO may be this depends on camd.sensor_fit)

    f_in_mm = camd.lens
    sensor_scale_x = fx / f_in_mm
    sensor_scale_y = fy / f_in_mm

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if camd.sensor_fit == 'VERTICAL':
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        camd.sensor_height = H / sensor_scale_y
        camd.sensor_width = W / sensor_scale_x / pixel_aspect_ratio
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        camd.sensor_width = W / sensor_scale_x
        camd.sensor_height = H * pixel_aspect_ratio / sensor_scale_y


def print_object_attributes(obj):
    """Helper to print object attributes"""
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))


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

    print_object_attributes(cam.data)

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
