@echo off
setlocal
cd /d "%~dp0"

set "MODE=%~1"
set "BASE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "PY_ARGS="
if exist "%BASE_PYTHON%" goto :check_python

set "BASE_PYTHON=py"
set "PY_ARGS=-3.12"
where py >nul 2>&1
if not errorlevel 1 (
    %BASE_PYTHON% %PY_ARGS% -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 else 1)" >nul 2>&1
    if not errorlevel 1 goto :check_python
)

where winget >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python 3.12 x64 hoac winget.
    echo Hay cai Python 3.12 x64 roi chay lai INSTALL.bat.
    pause
    exit /b 1
)

echo Dang cai Python 3.12 x64 cho tai khoan Windows nay...
winget install --id Python.Python.3.12 --exact --scope user --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :error
set "BASE_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
set "PY_ARGS="
if not exist "%BASE_PYTHON%" (
    echo [LOI] Khong tim thay Python312\python.exe sau khi cai.
    echo Hay mo terminal moi va chay lai INSTALL.bat.
    pause
    exit /b 1
)

:check_python
%BASE_PYTHON% %PY_ARGS% -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 else 1)"
if errorlevel 1 (
    echo [LOI] Du an can Python 3.12 x64.
    pause
    exit /b 1
)

set "ENV_PY=%~dp0.photobooth-env\Scripts\python.exe"
if not exist "%ENV_PY%" (
    echo Dang tao moi truong Python rieng...
    %BASE_PYTHON% %PY_ARGS% -m venv .photobooth-env
    if errorlevel 1 goto :error
)

if not exist ".photobooth-env\.photobooth-ready" goto :install_packages
goto :dispatch

:install_packages
echo Dang cai thu vien cho source va dong goi...
"%ENV_PY%" -m pip install --only-binary=:all: pillow==13.0.0 numpy==2.5.4 onnxruntime-directml==1.24.4 onnx==1.18.0 pyinstaller==6.22.3
if errorlevel 1 goto :error
type nul > ".photobooth-env\.photobooth-ready"

:dispatch
if /i "%MODE%"=="run" goto :run_source
if /i "%MODE%"=="build" goto :build_package
if not "%MODE%"=="" (
    echo Cach dung:
    echo   INSTALL.bat        Cai moi truong source
    echo   INSTALL.bat run    Cai neu can, sau do chay source worker
    echo   INSTALL.bat build  Cai neu can, sau do dong goi lai Photobooth
    pause
    exit /b 2
)

echo.
echo SETUP HOAN TAT.
echo Chay source: INSTALL.bat run
echo Dong goi lai: INSTALL.bat build
echo Model se tu tai o lan chay source dau tien va can Internet.
pause
exit /b 0

:run_source
"%ENV_PY%" -u "%~dp0src\remove_background.py" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:build_package
"%ENV_PY%" "%~dp0build_photobooth.py"
exit /b %ERRORLEVEL%

:error
echo.
echo [LOI] Cai dat hoac chay that bai. Hay kiem tra ket noi Internet va loi phia tren.
pause
exit /b 1
