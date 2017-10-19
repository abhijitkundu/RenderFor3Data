"""
Contains definition of ImageDataset class
"""

import os.path as osp
from collections import OrderedDict
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.6f')


class ImageDataset(object):
    """ImageDataset class
    Attributes:
        data: Contains image_infos, name, rootdir, meta_info
    """

    def __init__(self, name='ImageDataset', data=None):
        if data is None:
            self.data = OrderedDict()
            self.data['name'] = name
            self.data['rootdir'] = ''
            self.data['metainfo'] = {}
            self.data['image_infos'] = []
        else:
            self.data = data
            assert osp.exists(self.data['rootdir']), 'Root dir does not exist: {}'.format(self.data['rootdir'])

    def name(self):
        """Return dataset name"""
        return self.data['name']

    def rootdir(self):
        """Return dataset rootdir"""
        return self.data['rootdir']

    def image_infos(self):
        """Return dataset image_infos"""
        return self.data['image_infos']

    def num_of_images(self):
        """Return number of image_infos"""
        return len(self.data['image_infos'])

    def metainfo(self):
        """Return dataset metainfo"""
        return self.data['metainfo']

    def set_name(self, new_name):
        """Sets dataset name"""
        self.data['name'] = new_name

    def set_rootdir(self, rootdir):
        """Sets dataset rootdir"""
        self.data['rootdir'] = rootdir

    def set_image_infos(self, image_infos):
        """Sets dataset image_infos"""
        self.data['image_infos'] = image_infos

    def set_metainfo(self, meta_info):
        """Sets dataset metainfo"""
        self.data['metainfo'] = meta_info

    def add_image_info(self, image_info):
        """Add new image_info"""
        self.data['image_infos'].append(image_info)

    def write_data_to_json(self, filename):
        """Writes dataset to json"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2, separators=(',', ':'), cls=DatasetJSONEncoder)
        print('Saved dataset to {}'.format(filename))

    def set_data_from_json(self, filename):
        """Sets the dataset data from a JSON file"""
        with open(filename, 'r') as f:
            self.data = json.load(f, object_pairs_hook=OrderedDict)

    @classmethod
    def from_json(cls, filename):
        """Constricts the dataset from a JSON file"""
        with open(filename, 'r') as f:
            loaded_data = json.load(f, object_pairs_hook=OrderedDict)
        return cls(data=loaded_data)

    def __repr__(self):
        return 'ImageDataset(name="%s", with %d image_infos)' % (self.name(), self.num_of_images())


class NoIndent(object):

    """Helper class for preventing indention while json serialization
    Usage:
        json.dump(NoIndent([1, 2, 3]), file, indent=2, cls=DatasetJSONEncoder)
    """

    def __init__(self, value):
        self.value = value


class DatasetJSONEncoder(json.JSONEncoder):
    """Custom json decoder used by Dataset"""

    def default(self, o):
        if isinstance(o, NoIndent):
            return "@@" + repr(o.value).replace(' ', '').replace("'", '"') + "@@"
        return DatasetJSONEncoder(self, o)

    def iterencode(self, o, _one_shot=False):
        for chunk in super(DatasetJSONEncoder, self).iterencode(o, _one_shot=_one_shot):
            if chunk.startswith("\"@@"):
                chunk = chunk.replace("\"@@", '')
                chunk = chunk.replace('@@\"', '')
                chunk = chunk.replace('\\"', '"')
            yield chunk
