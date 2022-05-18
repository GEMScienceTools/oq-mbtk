@echo off
setlocal
set mypath=%~dp0
set PATH=%mypath%\python3.6;%mypath%\python3.6\Scripts;%PATH%
echo OpenQuake environment loaded
echo To see versions of installed software run 'pip freeze'
echo To run OpenQuake use 'oq' and 'oq engine'
cmd /k
endlocal
