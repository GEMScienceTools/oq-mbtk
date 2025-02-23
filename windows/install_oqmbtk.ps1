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

param (
    [string]$mbtk_branch = "master"  # default
)

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
Invoke-WebRequest -Uri https://ftp.openquake.org/windows/thirdparties/python-3.11.6-win64.zip -OutFile .\python.zip
Expand-Archive -Path .\python.zip -DestinationPath .\mbtk\python3 -Verbose
Remove-Item .\python.zip
cd $workDir\python3
Write-Host "Current Working Directory: $workDir"
Write-Host "Download and Installation of pip, wheel and setuptools"
Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile .\get-pip.py
$Env:PY_PIP = "$workDir\python3\Scripts"
$Env:PY_LIBS = "$workDir\python3\Lib;$workDir\python3\Lib\site-package"
$Env:PY_HOME = "$workDir\python3"
$Env:PATH="$workDir\python3\Scripts;$ENV:PATH"
$Env:PYTHONUTF8=1
Set-Alias -Name python -Value $Env:PY_HOME\python.exe
Set-Alias -Name pip -Value $Env:PY_PIP\pip.exe
#
python .\get-pip.py
#
cd $workDir
# TODO: make it possible to checkout a different branch instead of master
Write-Host "clone oq-engine and install it in developer mode"
git clone --depth=1 https://github.com/gem/oq-engine.git
cd .\oq-engine\
pip install -r .\requirements-py311-win64.txt
pip install -e .
cd ..
Write-Host "clone oq-mbtk on branch $mbtk_branch and install it"
git clone --depth=1 https://github.com/GEMScienceTools/oq-mbtk.git -b $mbtk_branch
cd .\oq-mbtk\
pip install -r .\requirements_win64.txt
pip install -e .
Write-Host "End of installation"
Write-Host "Creation of symlink for bat files on the Desktop of user $ENV:USERNAME"
cd windows
Copy-Item  -Path .\oq-console.cmd -Destination "$ENV:USERPROFILE\Desktop"
Copy-Item  -Path .\oq-server.cmd -Destination "$ENV:USERPROFILE\Desktop"
cd $workDir
