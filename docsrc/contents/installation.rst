Installation
============
The *oq-mbt* is installed with the procedure described in the following. 
Note that this procedure implies the installation of the OpenQuake engine. 
It was tested on Windows, Mac OS and Linux systems.

Here we demonstrate the installation of the MBTK for a linux system and Python 3.11.

* Open a terminal and move to the folder where you intend to install the tools;
* Upgrade pip:

```bash
python -m pip install --upgrade pip
```

* Install the OpenQuake engine and activate its virtual environment:

.. code-block:: bash

    $ git clone --depth=1 https://github.com/gem/oq-engine.git
    $ cd oq-engine
    $ python3 install.py devel
    $ source ~/openquake/bin/activate


* Go to the folder where you cloned the oq-mbtk repository and complete the
installation running the following commands,
making sure to replace `requirements-py311-linux.txt` with the name of
the file corresponding to the correct python version and operating system:

.. code-block:: bash

    $ pip install -e .
    $ pip install -r requirements-py311-linux.txt
