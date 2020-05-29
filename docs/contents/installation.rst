Installation
============
The *oq-mbt* is installed with the procedure described in the following. 
Note that this procedure implies the installation of the OpenQuake engine. 
It was tested on Mac OS and Linux systems.

* Open a terminal and move to the folder where to intend to install the tools;
* Create a virtual environment with ``python3 -m venv venv``
* Activate the virtual environment ``source venv/bin/activate``
* Update pip ``pip install -U pip``
* Enter the virtual environment ``cd venv`` and create a directory for storing source code ``mkdir src; cd src``
* Clone the OpenQuake engine ``git clone git@github.com:gem/oq-engine.git``
* Complete a development installation with ``cd ..`` then ``pip install -r ./src/oq-engine/requirements-py36-macos.txt`` and finally ``pip install -e ./src/oq-engine/``
