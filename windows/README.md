# OpenQuake Model Building Toolkit on Windows
The OpenQuake Model Building Toolkit (OQ-MBTK) can be installed on Windows 
with powershell installer.

To install the OQ-MBTK on Windows 10 run the following command from the PowerShell (not the normal CMD command prompt):

```
C:\>curl.exe -LO https://raw.githubusercontent.com/GEMScienceTools/oq-mbtk/master/windows/install_oqmbtk.ps1
C:\>.\install_oqmbtk.ps1 
```

This will install the OQ-MBTK software and OpenQuake Engine in a new folder called 'mbtk' under the USER home directory: $ENV:USERPROFILE\mbtk 
and create two cmd files on the Desktop to load the OQ-MBTK environment and Openquake environment

- this Installer includes its own distribution of the dependencies needed 
    - Python 3.8.10
    - Python dependencies (pip, numpy, scipy, and more)

To use the environment just activate from the command files on the desktop:

- oq-console.cmd: To load the  OQ-MBTK and OpenQuake Engine environment
- oq-server.cmd: To start db server and webui of Openquake Engine

To uninstall the OQ-MBTK simply remove the mbtk folder and the cmd files on the desktop


## Requirements

Requirements are:

- Windows 10 (64bit)
- 4 GB of RAM (8 GB recommended)
- 1.5 GB of free disk space

**Windows 7** and **Windows 8** are not supported. That means that we do
not test such platforms and the openquake model building toolkit may or may not work there. 
