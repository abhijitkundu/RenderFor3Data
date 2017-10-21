import math

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
    R = R_bcam2cv * R_world2bcam
    t = R_bcam2cv * T_world2bcam

    # put into 3x4 matrix
    Rt = Matrix((R[0][:] + (t[0],),
                 R[1][:] + (t[1],),
                 R[2][:] + (t[2],)))
    return Rt


def set_blender_camera_extrinsic(cam, R, t):
    """
    Move blender camera to have the provided extrinsic of [R|t]
    """
    # bcam stands for blender camera
    R_cv2bcam = Matrix(((1,  0,  0),
                        (0, -1,  0),
                        (0,  0, -1)))

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2bcam = R_cv2bcam * R
    t_world2bcam = R_cv2bcam * t

    R_bcam2world = R_world2bcam.transposed()
    t_bcam2world = -1 * R_bcam2world * t_world2bcam

    cam.matrix_world = Matrix.Translation(t_bcam2world) * R_bcam2world.to_4x4()

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


def wrap_to_pi(radians):
    """Wrap an angle (in radians) to [-pi, pi)"""
    # wrap to [0..2*pi]
    wrapped = radians % (2 * math.pi)
    # wrap to [-pi..pi]
    if wrapped >= math.pi:
        wrapped -= 2 * math.pi

    return wrapped


def is_rotation_matrix(R, atol=1e-6):
    """Checks if a matrix is a valid rotation matrix"""
    assert len(R.row) == 3, "R is not a 3x3 matrix. R = {}".format(R)
    assert len(R.col) == 3, "R is not a 3x3 matrix. R = {}".format(R)
    Rt = R.transposed()
    shouldBeIdentity = Rt * R
    I = Matrix.Identity(3)
    delta = I - shouldBeIdentity
    sq_norm = 0.0
    for i in range(3):
        for j in range(3):
            sq_norm += (delta[i][j] ** 2)
    return math.sqrt(sq_norm) < atol


def skew_symm(x):
    """Returns skew-symmetric matrix from vector of length 3"""
    assert len(x) == 3, "x is not vector of length 3 since x = {}".format(x)
    return Matrix(((0, -x[2], x[1]),
                   (x[2], 0, -x[0]),
                   (-x[1], x[0], 0)))


def rotation_from_two_vectors(a, b):
    """
    Returns rotation matrix that rotates a to be same as b
    See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    """
    assert len(a) == 3
    assert len(b) == 3
    au = a.normalized()
    bu = b.normalized()
    ssv = skew_symm(au.cross(bu))
    c = au.dot(bu)
    assert c != -1.0, 'Cannot handle case when a and b are exactly opposite'
    inv_one_plus_c = 1. / (1. + c)
    R = Matrix.Identity(3) + ssv + inv_one_plus_c * (ssv * ssv)
    assert is_rotation_matrix(R)
    return R


def rotationX(angle):
    """Get the rotation matrix for rotation around X"""
    c = math.cos(angle)
    s = math.sin(angle)
    Rx = Matrix(((1, 0,  0),
                 (0, c, -s),
                 (0, s,  c)))
    return Rx


def rotationY(angle):
    """Get the rotation matrix for rotation around X"""
    c = math.cos(angle)
    s = math.sin(angle)
    Ry = Matrix(((c,  0, s),
                 (0,  1, 0),
                 (-s, 0, c)))
    return Ry


def rotationZ(angle):
    """Get the rotation matrix for rotation around X"""
    c = math.cos(angle)
    s = math.sin(angle)
    Rz = Matrix(((c, -s, 0),
                 (s,  c, 0),
                 (0,  0, 1)))
    return Rz


def rotation_from_viewpoint(vp):
    """Get rotation matrix from viewpoint [azimuth, elevation, tilt]"""
    assert len(vp) == 3
    assert -math.pi <= vp[0] <= math.pi
    assert -math.pi / 2 <= vp[1] <= math.pi / 2
    assert -math.pi <= vp[2] <= math.pi

    R = rotationZ(-vp[2] - math.pi / 2) * rotationY(vp[1] + math.pi / 2) * rotationZ(-vp[0])
    assert is_rotation_matrix(R)
    return R
