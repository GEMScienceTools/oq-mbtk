@echo off
setlocal

set mypath=%USERPROFILE%\mbtk
set PATH=%mypath%;%mypath%\Scripts;%PATH%
set PY_PIP=%mypath%\Scripts
set PY_LIBS=%mypath%\Lib;%mypath%\Lib\site-package
set PY_HOME=%mypath%
set PYTHONUTF8=1
echo OpenQuake environment loaded
echo To see versions of installed software run 'pip freeze'
echo To run OpenQuake use 'oq' and 'oq engine'
cmd /k

endlocal
