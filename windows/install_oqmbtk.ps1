#
# OQ-MBTK Windows 10 Installation Script
#
# This script MUST be executed using PowerShell (not cmd)
#
# Copyright (C) 2022 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
#
cd $ENV:USERPROFILE
$curDir = Get-Location
$workDir = $curDir.Path + "\mbtk"
# Test to see if folder [$workDir]  exists
if (Test-Path -Path $workDir) {
    Write-Host "Path $workDir exist."
    Write-Host "We can not install the OQ-MBTK environment in the folder $workDir"
	EXIT 1
} else {
    Write-Host "Path $workDir doesn't exist."
    Write-Host "We can install the OQ-MBTK environment in the folder $workDir"
}
Write-Host "Current Working Directory: $workDir"
Write-Host "Download and Installation of python embeddable package"
Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.8/python-3.10.8-embed-amd64.zip -OutFile .\python.zip
Expand-Archive -Path .\python.zip -DestinationPath .\mbtk -Verbose
Remove-Item .\python.zip
cd $workDir
Write-Host "Current Working Directory: $workDir"
Write-Host "Download and Installation of pip, wheel and setuptools"
Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile .\get-pip.py
$Env:PY_PIP = "$workDir\Scripts"
$Env:PY_LIBS = "$workDir\Lib;$workDir\Lib\site-package"
$Env:PY_HOME = "$workDir"
$Env:PATH="$workDir\Scripts;$ENV:PATH"
$Env:PYTHONUTF8=1
Set-Alias -Name python -Value $Env:PY_HOME\python.exe
Set-Alias -Name pip -Value $Env:PY_PIP\pip.exe
#
python .\get-pip.py
Move-Item .\python310._pth .\python310._pth.old
#
pip install pytest
Write-Host "clone of the branch $branch for oq-engine and install in developer mode"
git clone --depth=1 https://github.com/gem/oq-engine.git
cd .\oq-engine\
pip install -r .\requirements-py310-win64.txt
pip install -e .
cd ..
Write-Host "clone of the branch $branch for oq-mbtk and install in developer mode"
git clone --depth=1 https://github.com/GEMScienceTools/oq-mbtk.git
cd .\oq-mbtk\
pip install pandas==1.4.0
pip install pyzmq==24
pip install jupyter-server==2.6.0
pip install xarray==2023.5.0
pip install -r .\requirements_win64.txt
pip install -e .
Write-Host "End of installation"
Write-Host "Creation of symlink for bat files on the Desktop of user $ENV:USERNAME"
cd windows
Copy-Item  -Path .\oq-console.cmd -Destination "$ENV:USERPROFILE\Desktop"
Copy-Item  -Path .\oq-server.cmd -Destination "$ENV:USERPROFILE\Desktop"
cd $workDir
