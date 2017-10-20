#!/bin/bash

# SET PATHS HERE
BLENDER_PATH=/home/abhijit/SoftwareInstalls/Blender/blender-2.78a-linux-glibc211-x86_64

# Uses python3 because of Blender
$BLENDER_PATH/blender --background --python camera_calibration_test.py
