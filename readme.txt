Required packages:

plotly
stable_baselines3 extra
optuna
pettingzoo

NOTES: 
-dont install rlcard as this is already included with required modifications 
-made using VScode, suggested to run in this IDE to assist with compatibility
- filepaths to save figures and trained models will likely have to be replaced with a filepath to a folder on your own system. 

Instructions:
- the 'project_main' file collects all the required python files used to fulfill the project.
  The sections do not appear in sequential order, but in the order of that chosen in the dissertation.
 The first part relate to the project design and offer quick access to the corresponding files. please open in debugging mode and 
change JSON setting to "justMyCode": false. this will alllow easy access to the files. The second part relate to the project implementation
. ie, how to code was used to generate the results and findings used in the project. Finally, the third part include demonstrations:
the exact same code as in the implementation and results section but with the variables changed to smaller values 
to demonstrate the code runs successfully.

- when running for the first time. the 'path_gym' function of _patch_env will raise an exception saying the environment is not recognised as an open AI gymnasium environment.
this is caused by the custom wrapper modifications making it not being recognised, despite being fully compatible. You can either delete this file on your device and replace it with 
the modified patch_gym.py file included in this folder. Alternativley you can easily modify the _patch_env function by hashing out all the code and just place 'return env'.


Regards - Rhys 
