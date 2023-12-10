import os

import ml_champ.core.configuration as ai_config
from ml_champ.core.historical import History_Record

'''
    utils.__init__.py:
    The methods in here are commonly needed in order to more easily interact with the 
    ml_champ APIs.  You are free to add to the methods but it is advisable to 
    keep the original methods that come with the default version of this file.
'''
def get_parent_directory(path, levels=1):
    parent = path
    for i in range(levels):
        parent = os.path.dirname(parent)        
    return parent

def get_project_directory():
    # The project directory is assumed to be two levels above this file.
    project_folder_parent = get_parent_directory(os.path.abspath(__file__), levels=2)
    return project_folder_parent

'''
    This method gets called by the ml_champ Project's run_custom_script() method
    allowing you to create customized processes that can be called by the Project's interface.
    For example, if you wanted to get the model_tasks.get_trained_model() output, you would
    import the model_tasks method, retrieve it and pass it up if that is what you need.
'''
def run_custom_script(script_name):
    print('Running custom script', script_name)
    