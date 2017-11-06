#!/usr/bin/python3

import numpy as np
import json
from collections import OrderedDict
from RenderFor3Data.image_dataset import ImageDataset, NoIndent, DatasetJSONEncoder

def main():
    """Main Function"""
    K = np.array([[721.5377, 0.0, 621.0], [0.0, 721.5377, 187.5], [0.0, 0.0, 1.0]])
    K_inv = np.linalg.inv(K)
    image_size = [1242, 375]
    vp = np.array([1.57079632679, 0.01265, 0.010868])
    dimension = np.array([3.691882, 1.936846, 1.587618])
    shape_file = "CityShapes/Cars/Audi_A4/Audi_A4.obj"
    image_center = np.array([160.0, 187.5])

    image_info = OrderedDict()
    image_info['image_intrinsic'] = NoIndent(K.tolist())
    image_info['image_size'] = NoIndent(image_size)
    image_info['image_file'] = "SameViewpoint.png"

    obj_infos = []
    num_of_cars = 5
    step_x = (image_size[0] - 2 * image_center[0]) / (num_of_cars - 1)
    for i in range(num_of_cars):
        obj_info = OrderedDict()
        obj_info['id'] = i + 1
        obj_info['shape_file'] = shape_file
        obj_info['dimension'] = NoIndent(dimension.tolist())
        obj_info['viewpoint'] = NoIndent(vp.tolist())
        center_proj = image_center + np.array([i * step_x, 0])
        obj_info['center_proj'] = NoIndent(center_proj.tolist())

        center_proj_ray = K_inv.dot(np.append(center_proj, 1))
        center_proj_ray = center_proj_ray / center_proj_ray[2] * 16.0
        obj_info['center_dist'] = np.linalg.norm(center_proj_ray)
        obj_infos.append(obj_info)

    image_info['object_infos'] = obj_infos

    with open("SameViewpoint.json", 'w') as f:
        json.dump(image_info, f, indent=2, separators=(',', ':'), cls=DatasetJSONEncoder)


if __name__ == '__main__':
    main()
