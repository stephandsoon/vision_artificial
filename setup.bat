@echo off

echo.
echo.
echo **************************************************************************
echo * Setup python and paths
echo **************************************************************************

:: Default location of Python 3.8
set PYTHON= "C:\Users\steph\AppData\Local\Programs\Python\Python310\python.exe"
:: or wherever your python.exe is on your system (i. e. "C:\Program Files\Python38\python.exe")

if not exist %PYTHON% (
	echo ERROR: Python not found under %PYTHON%
	goto end
) else (
	echo Python found at %PYTHON%
)


echo.
echo.
echo **************************************************************************
echo * Create virtual environment
echo **************************************************************************

:: Default location of virtual environment is in .venv
:: %~dp0 gives current directory of bat-file

set VENV_PATH=%~dp0.venv

if not exist %VENV_PATH% (
	%PYTHON% -m venv %VENV_PATH%
	echo Creating virtual environment at %VENV_PATH%
) else (
	echo Virtual environment already existing at %VENV_PATH%
)

	
echo.
echo.
echo **************************************************************************
echo * Install required packages
echo **************************************************************************

:: Setup pip command 
set PIP=%VENV_PATH%\Scripts\pip.exe --trusted-host pypi.org

%PIP% install --upgrade pip
%PIP% install -r requirements.txt	


:end
echo.
echo.
pause