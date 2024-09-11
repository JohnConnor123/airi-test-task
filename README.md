# Description
This library contains a complete development pipeline for performing an AIRI recruitment test task. 
Briefly, the goal of the problem is to predict the `DDG` value (change in protein stability due to mutations). 
A complete description of the test task can be found in the Jupyter notebook located at `src/notebooks/test_task.ipynb`.

# Installation:
### From GitHub
```
git clone https://github.com/JohnConnor123/airi-test-task.git
cd airi-test-task
poetry shell
cd src
source ./mlflow-init-commands.sh
python train.py
```
### From PyPI:
First, you need to download the `mlflow-init-commnads.sh` script in the `src` folder to start the mlflow server.
After that just do:
```
poetry add airi-test-task
```
