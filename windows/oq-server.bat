@echo off
setlocal
set mypath=%~dp0
set PATH=%mypath%\python3.6;%mypath%\python3.6\Scripts;%PATH%
set OQ_HOST=localhost
set OQ_PORT=8800


echo Starting the server.
echo Please wait ...
REM Start the WebUI using django
oq webui start %OQ_HOST%:%OQ_PORT%

endlocal
exit /b 0
