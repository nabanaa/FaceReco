# install virtualenv command
pip install --uprgrade pip
pip install -r "requirements.txt"
# setup and activate the environment
if (virtualenv --python=3.11.1 .env -and source .env/bin/activate)
{
    pip install -r "../requirements.txt"
}

 