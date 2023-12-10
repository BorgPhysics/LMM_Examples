from __future__ import absolute_import

from ..config import *
from ..utils import *

from ml_champ.core.historical import History_Record

# Connect to your project's code
from ..qa_pipeline.main import analysis_processor

history_record = None
pretraining_commands = []

'''
    Do Not Edit this method, duplicate it or otherwise reference it where it doesn't already exist in this file.
    If a method does not have access to the history record, that is by design.
'''
def get_history_record(run_id=None, run_name=None):
    global history_record
#     print('model_tasks.history_record:', history_record)
    if not history_record:
        # Need to build it so that you have the datestamp
        project_directory = get_project_directory()
        history_record = History_Record(project_directory=project_directory, run_id=run_id, run_name=run_name)
        
    return history_record

"""
    run_preprocessing(preprocessing_commands)
    :param preprocessing_commands: JSON formatted string that is used to pass in commands prior to training.
"""
def run_preprocessing(preprocessing_commands): 
    global pretraining_commands
    pretraining_commands.append(preprocessing_commands)
    
    # This code is customized according to the needs of your particular project.
    # If you need to inject the location of a pre-trained model, this is where it should occur.
    # 
    # If you want to load a pretrained model from another run, you must use the pretrained_model_location variable
    # to specify the location of the parent Experiment/Run directory.  Do not use a directory location below 
    # the UUID-style run directory.  This method should then implement the code necessary to retrieve and 
    # load the pretrained model from any subdirectory from there (typically below the Run's archives directory).
    # 
    # This method can be used multiple times before training but should not be called after that.
    
    # Example of updating the pretrained_model_location path with the artifacts folder prior to passing on to your project.
    # The incoming pretrained_model_location has to point to the Run directory - not the artifacts directory
    pretrained_model_location = preprocessing_commands.get('pretrained_model_location', None)
    if pretrained_model_location:
        # We are modifying the location to include artifacts here 
        # so that always needing an artifacts directory isn't forced upon the project.
        preprocessing_commands['pretrained_model_location'] = os.path.join(pretrained_model_location, 'artifacts')
        
    # For now, I'm going to use the preprocessing commands to test functionality in the code without generating artifacts
    analysis_processor.run_various_components()
    analysis_processor.run_preprocessing_commands(pretraining_commands)
    
"""
    train_model()
    This method should handle training, testing and saving the trained model.
    You may break these out into separate methods but an ML CHAMP method is not provided to run them separately.
    This is by design since many features of ML CHAMP could not be achieved otherwise.
"""
def train_model():
    ###############################
    #  DO NOT EDIT
    history_record = get_history_record()
    ############################### 
    
    ###############################
    #  Your code begins here....
    #  Implement the following hooks to your code:
    #  - Training.
    #  - Generating test results.
    #  - Save the trained model.
    ###############################
    pass

"""
    run_postprocessing(preprocessing_commands)
    :param postprocessing_commands: JSON formatted string that is used to pass in commands prior to training.
"""
def run_postprocessing(postprocessing_commands):
    # This code is customized according to the needs of your particular project.
    # These commands are designed to be run after your model has been trained.
    # For example, this code block could be designed to return your trained model, the training data that was used, etc.
    # in order to supply this information to another application that generates advanced statistics but only if certain
    # thresholds were surpassed.  While you could theoretically achieve this with a combination of preprocessing and 
    # train_model code, this method allows you more fine-grained manual control if needed.
        
    return None
     
"""
    test_model()
    This method assumes that training has already been run.  However, it does not know if you are calling it from
    the same in-memory instance that generated the trained model or if it's being called from a previously trained
    snapshot in an archive directory.  In other words, you cannot assume that train_model has been run in the current
    session or that any variables that method generates will exist when you call this method.  
    It is up to you to make sure that this method knows how to load the model from the file system if it isn't available.
    Never attempt to handle this by attempting to call train_model() from here.
"""
def test_model():
    global history_record
    if not history_record:
        # Training was not run in this session.  Your code should handle loading the variables that it needs here.
        # Note that since a history record does not exist, you cannot (and should not) attempt to generate the history record
        # or pass metrics to it.  If the history record exists, then you can assume that this is the first case and it is
        # safe to log metrics in the history record like normal.
        pass
    pass