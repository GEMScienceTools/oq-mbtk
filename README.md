# oq-mbtk
OpenQuake Model Building Toolkit

[![Build Status](https://travis-ci.org/GEMScienceTools/oq-mbtk.svg?branch=master)](https://travis-ci.org/GEMScienceTools/oq-mbtk)

## Development installation
```
# default paths and venv
MY_VENV=~/.virtualenvs/mbtk
MY_BASEDIR=~/base_dir

# creation of virtual environment
python3.5 -m venv $MY_VENV
~/.virtualenvs/mbtk/bin/activate
install -U pip

# creation of a base directory
mkdir -p "$MY_$BASEDIR"
cd "$MY_$BASEDIR"

# preventively installation of GDAL
pip install "http://ftp.openquake.org/wheelhouse/linux/py35/GDAL-2.2.4-cp35-cp35m-manylinux1_x86_64.whl"

git clone "git@github.com:gem/oq-engine.git"
pip install -e oq-engine

git clone "git@github.com:GEMScienceTools/oq-subduction.git"
pip install -e oq-subduction

git clone "git@github.com:GEMScienceTools/oq-mbtk.git"
pip install -e oq-mbtk

```
