$branch = master
$curDir = Get-Location
$workDir = $curDir.Path + "\mbtk"
Write-Host "Current Working Directory: $curDir"
Write-Host "Download and Installation of python embeddable package"
Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.8.10/python-3.8.10-embed-amd64.zip -OutFile .\python.zip
Expand-Archive -Path .\python.zip -DestinationPath .\mbtk -Verbose
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
Move-Item .\python38._pth .\python38._pth.old
#
pip install pytest
git clone -b $branch --depth=1  https://github.com/gem/oq-engine.git
cd .\oq-engine\
pip install -r .\requirements-py38-win64.txt
pip install -e .
cd ..
git clone -b $branch https://github.com/GEMScienceTools/oq-mbtk.git
cd .\oq-mbtk\
pip install -r .\requirements_win64.txt
pip install -e .
