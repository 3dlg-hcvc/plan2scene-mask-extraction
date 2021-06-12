import json

class Config:
    def __init__(self, d):
        for k,v in d.items():
            if isinstance(v, dict):
                v = Config(v)
            self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]


def parse_config(config_path):
    """
    Parses a json config file into a Config object.
    """
    with open(config_path, "r") as f:
        config_dict = json.loads(f.read())
    if isinstance(config_dict, dict):
        return Config(config_dict)
    else:
        return config_dict

