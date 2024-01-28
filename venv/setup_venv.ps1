# install virtualenv command
pip install -r
# setup and activate the environment
virtualenv --no-site-packages --python=3.11.1 --distribute .env && source .env/bin/activate && pip install -r ../requirements.txt 