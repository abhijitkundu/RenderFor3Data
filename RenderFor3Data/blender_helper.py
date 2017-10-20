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