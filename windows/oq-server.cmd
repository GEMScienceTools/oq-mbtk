@echo off
setlocal
set mypath=%USERPROFILE%\mbtk
set PATH=%mypath%;%mypath%\Scripts;%PATH%
set PY_PIP=%mypath%\Scripts
set PY_LIBS=%mypath%\Lib;%mypath%\Lib\site-package
set PY_HOME=%mypath%
set PYTHONUTF8=1
set OQ_HOST=localhost
set OQ_PORT=8800

echo OpenQuake environment loaded
echo Starting the server.
echo Please wait ...
REM Start the WebUI using django
oq webui start %OQ_HOST%:%OQ_PORT%

endlocal
exit /b 0
