#!/bin/bash

IMAGE_INFO_FILE_PATH=${1}

# SET PATHS HERE
BLENDER_EXEC=blender

# Uses python3 because of Blender
$BLENDER_EXEC --background -t 1 --python render_single_image_info.py -- ${IMAGE_INFO_FILE_PATH}
