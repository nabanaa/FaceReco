# install virtualenv command
pip install --upgrade pip
pip install -r "requirements.txt"
# setup and activate the environment
if (virtualenv --python=3.11.1 env -and ./env/bin/activate)
{
    pip install -r "../requirements.txt"
}
./env/bin/activate
pip install --upgrade pip
 