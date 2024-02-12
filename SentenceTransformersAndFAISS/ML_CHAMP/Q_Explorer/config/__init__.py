import ml_champ.core.configuration as ai_config
from ..utils import *

config_file = None

def get_config():
    """
    :return: The current config file read at request time.
    It needs to be reloaded each time that it's called in case it's been modified...
    However, it should only need to be reloaded by the methods in the tasks directory.
    """
    global config_file
    if not config_file:
        load_config()
            
    return config_file

'''
    load_config() forces a reload of the config file.  
    Should only be directly called by methods in the tasks directory.
'''
def load_config():
    global config_file    
    project_directory = get_project_directory()
    config_file = ai_config.get_config(project_directory)       
    
'''
    Use the version from ml_champ that will have failover implemented.
'''
def get_config_value(key_path):
    global config_file
    if not config_file:
        config_file = get_config()
    return ai_config.get_config_value(config_file, key_path)