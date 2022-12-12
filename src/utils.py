import yaml


def read_config_file(fpath):
    """Load the project config yaml file
    args:
        fpath(str): path to the config file
    """

    config = None
    try: 
        with open(fpath, 'r') as f:
            config = yaml.safe_load(f)
    except:
        raise FileNotFoundError('Couldnt load the file')
    
    return config
    