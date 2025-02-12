# OpenQuake Model Building Toolkit on Windows
The OpenQuake Model Building Toolkit (OQ-MBTK) can be installed on Windows
with the PowerShell installer.

## Requirements

Requirements are:

- Windows 10 or Windows 11
- 8 GB of RAM (16 GB recommended)
- 1.5 GB of free disk space
- Git on Windows
  The most official build is available for download on the Git website. Just go to https://git-scm.com/download/win and the download will start automatically.

**Windows 7** and **Windows 8** are not supported. That means that we do
not test such platforms and the openquake model building toolkit may or may not work there. 

## PowerShell Execution Policy

The Powershell execution policy is a rule that defines which scripts are allowed to run on a specific server or workstation.
The default execution policy is Restricted, so no scripts are allowed to run.
If you want to check which execution policy is currently configured, you can use the Get-ExecutionPolicy cmdlet.

```
Get-ExecutionPolicy
```

You can change to a new execution policy with the Set-ExecutionPolicy cmdlet.

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Installation

To install the OQ-MBTK on Windows run the following command from the Windows Terminal :

```
C:\>curl.exe -LO https://raw.githubusercontent.com/GEMScienceTools/oq-mbtk/master/windows/install_oqmbtk.ps1
C:\>.\install_oqmbtk.ps1 
```

This will install the OQ-MBTK software and OpenQuake Engine in a new folder called 'mbtk' under the USER home directory: $ENV:USERPROFILE\mbtk 
and create two cmd files on the Desktop to load the OQ-MBTK environment and Openquake environment

- this Installer includes its own distribution of the dependencies needed 
    - Python 3.11.7
    - Python dependencies (pip, numpy, scipy, and more)

To use the environment just activate from the command files on the desktop:

- oq-console.cmd: To load the  OQ-MBTK and OpenQuake Engine environment
- oq-server.cmd: To start db server and webui of Openquake Engine

To uninstall the OQ-MBTK simply remove the mbtk folder and the cmd files on the desktop
