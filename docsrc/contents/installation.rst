Installation
============
The *oq-mbtk* is installed with the procedure described in the following. 
Note that this procedure implies the installation of the OpenQuake engine. 
It was tested on Windows, Mac OS and Linux systems.

Here we demonstrate the installation of the MBTK for a linux system and Python 3.11.

* Open a terminal and move to the folder where you intend to install the tools;
* Upgrade pip:

.. code-block:: bash

    $ python -m pip install --upgrade pip

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


Julia and PSHAModelBuilder
--------------------------

Some of the functions of the mbtk (especially the :code:`wkf` module) require the use of the
`PSHAModelBuilder <https://github.com/GEMScienceTools/PSHAModelBuilder>`_. These scripts
are written in Julia to provide a more computationally efficient approach for fixed
kernel smoothing and propogating rates to a smoothed model. To use the 
PSHAModelBuilder, install `Julia <https://julialang.org/>`_ and then do the following:

.. code-block:: bash
    
    $ julia
    $ ]
    $ add https://github.com/GEMScienceTools/PSHAModelBuilder.git  

GMT and pyGMT for plotting
--------------------------

Some of the functions in the mbtk will create maps with `the Generic Mapping Tools <https://www.generic-mapping-tools.org/>`_,
either directly or through the pyGMT package. You can find guidance on downloading `GMT <https://docs.generic-mapping-tools.org/latest/install.html>`_ or `pyGMT <https://www.pygmt.org/latest/install.html>`_ at the provided links. **Note** that although pyGMT is a requirement of the mbtk, you will need to install GMT in your virtual environment seperately following the above link if you wish to use the pyGMT plotting features. This site also contains helpful information on the most common installation issues (i.e. if you see an 'Error loading GMT shared library' error).