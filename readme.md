MRI processing pipeline

File list: <br />
train.py - wrapper to begin training of multi-class 3D MRI brain classifier. <br />
dataset.py - dataset manager, for loading, processing, saving the MRI images on the filesystem <br />
transforms.py - pre-processing transforms to apply to a MRI image <br />
utils.py - project utilities (i.e., project path manager to manage project paths, etc.) <br />
visualize.py - 2D and 3D visualization of MRI brain <br />
config.json - configuration file to select transforms <br />
hyperparams.json - configuration file to select CNN parameters <br />

To run the processing pipeline: <br />

1. The project folder structure should be maintained
   Project/ <br />
   |-- data/ <br />
   | |-- _data files_ <br />
   | <br />
   |-- env/ <br />
   | |-- _environment files_ <br />
   | <br />
   |-- src/ <br />
   | |-- _data files_ <br />
   | | |-- dataset.py <br />
   | | |-- models.py <br />
   | | |-- transforms.py <br />
   | | |-- utils.py <br />
   | | |-- visualize.py <br />
   | | |-- train.py <br />
   | | <br />
   | <br />
   |-- config.json <br />
   |-- hyperparams.json <br />
   |-- requirements.txt <br />
   |-- README <br />
2. Download project source files<br />
   The source files can be cloned or downloaded from:
   https://github.com/Jokubaskaralius/3D-MRI-Processing-pipeline

3. Download the dataset and store it in the project folder <br />
   The dataset can be downloaded from:
   https://www.kaggle.com/jokubaskaralius/3d-mri-t1we-brain-volumes <br />
   Place it according to staging step 1.

4. Activate your python virtualenv <br />
   Refer to creating a virtualenv and activating one <br />
   https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/ <br />
   This step may be skipped if this project and dependencies are to be installed globally,
   however virtualenv is highly recommended.

5. Install the project dependencies <br />
   Execute command: <br />
   _pip install -r requirements.txt_

6. Run the process.py wrapper <br />
   Execute command: <br />
   _python3 src/train.py_
