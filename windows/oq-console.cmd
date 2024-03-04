@echo off
setlocal

set mypath=%USERPROFILE%\mbtk
set PATH=%mypath%;%mypath%\python3\Scripts;%PATH%
set PY_PIP=%mypath%\python3\Scripts
set PY_LIBS=%mypath%\python3\Lib;%mypath%\python3\Lib\site-package
set PY_HOME=%mypath%
set PYTHONUTF8=1
echo OpenQuake environment loaded
echo To run OpenQuake use 'oq' and 'oq engine'
cd %mypath%
cmd /k

endlocal
