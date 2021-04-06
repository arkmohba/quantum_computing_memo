import yaml


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as yml:
            config = yaml.load(yml)
        self.model_obj_path = config["model_obj_path"]
        self.model_scene_path = config["model_scene_path"]
        self.sqlite_path = config["sqlite_path"]
        self.ngt_obj_path = config["ngt_obj_path"].encode('utf-8')
        self.ngt_scene_path = config["ngt_scene_path"].encode('utf-8')

        self.images_dir = config["images_dir"]
        self.tumnail_dir = config["tumnail_dir"]
        self.static_dir = config["static_dir"]
