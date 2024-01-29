#!/usr/bin/bash

# install virtualenv command
pip install --upgrade pip
pip install -r "requirements.txt"
# setup and activate the environment
virtualenv --python=3.11.1 env && source env/bin/activate && pip install -r "../requirements.txt" 
source env/bin/activate
pip install --upgrade pip
