# OpenQuake Model Building Toolkit [oq-mbtk]
The OpenQuake Model Building Toolkit is a container for various packages with unique modelling or analyses capabilities. The project started with the `mbt` package which provides tools for the construction of components of a PSHA earthquake occurrence model. 

Documentation accessible at [https://gemsciencetools.github.io/oq-mbtk/index.html](https://gemsciencetools.github.io/oq-mbtk/index.html)

![workflow](https://github.com/GEMScienceTools/oq-mbtk/tree/master/.github/workflows/test_deploy.yaml/badge.svg)

## Development installation
```
# default paths and venv
MY_VENV=~/.virtualenvs/mbtk
MY_BASEDIR=~/base_dir

# creation of virtual environment
python3.6 -m venv $MY_VENV
. "$MY_VENV/bin/activate"
pip install -U pip

# creation of a base directory
mkdir -p "$MY_BASEDIR"
cd "$MY_BASEDIR"

git clone "git@github.com:GEMScienceTools/oq-mbtk.git"

# this sed magic extract 'install' target of .travis.yml, reconstruct splitted lines and run them
cat  oq-mbtk/.travis.yml | \
sed '1,/^install:$/d;/^ *$/,$d' | \
sed ':a;N;/\n  - /!s/\n/ /;ta;P;D' | \
sed 's/ \&\&     / \&\& /g;s/^  - //g;s/^\(pip install -e .\)$/cd oq-mbtk ; \1/g' | \
bash

cd oq-mbtk
```

### how to run tests:
```
pytest -vsx openquake
```
