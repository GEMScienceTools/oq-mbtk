@echo off
setlocal

set mypath=%USERPROFILE%\mbtk
set PATH=%mypath%;%mypath%\Scripts;%PATH%
echo OpenQuake environment loaded
echo To see versions of installed software run 'pip freeze'
echo To run OpenQuake use 'oq' and 'oq engine'
cmd /k

endlocal
