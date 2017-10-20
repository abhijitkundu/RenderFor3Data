#!/bin/bash

# SET PATHS HERE
BLENDER_PATH=/home/abhijit/SoftwareInstalls/Blender/blender-2.78a-linux-glibc211-x86_64

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.4:${BUNDLED_PYTHON}/lib/python3.4/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# Uses python3 because of Blender
$BLENDER_PATH/blender --background --python camera_calibration_test.py
