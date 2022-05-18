# OpenQuake Model Building Toolkit on Windows
The OpenQuake Model Building Toolkit (OQ-MBTK) can be installed on Windows 
with powershell installer.
You actually need to invoke PowerShell from Command Prompt and from that window run:
```
C:\>curl.exe -LO https://raw.githubusercontent.com/GEMScienceTools/oq-mbtk/master/windows/install_oqmbtk.ps1
C:\>install_oqmbtk.ps1 
```

- this Installer includes its own distribution of the dependencies needed 
    - Python 3.8.10
    - Python dependencies (pip, numpy, scipy, and more)

## Requirements

Requirements are:

- Windows 10 (64bit)
- 4 GB of RAM (8 GB recommended)
- 1.5 GB of free disk space

**Windows 7** and **Windows 8** are not supported. That means that we do
not test such platforms and the openquake engine may or may not work there. 
