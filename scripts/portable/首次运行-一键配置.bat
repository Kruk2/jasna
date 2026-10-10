@echo off
rem ===========================================================
rem  Jasna portable package - first run, one click
rem
rem  Double-click this file. It will:
rem    1) configure the bundled python venv
rem    2) check / install the ROCm GPU kernels for your card
rem    3) set up the MIGraphX detection runtimes (uses the
rem       copy shipped inside the package, so no C: drive or
rem       Microsoft Store dependency)
rem    4) run a real detection-engine self test
rem
rem  All steps are repeatable; finished ones are skipped.
rem
rem  Optional switches are passed straight through:
rem      -CheckOnly     report only, change nothing
rem      -SkipRocm      do not touch the ROCm device packages
rem      -SkipSelfTest  do not load any model
rem  Example:
rem      (right click - Run with PowerShell, or)
rem      first_run_setup.bat -CheckOnly
rem ===========================================================
setlocal
chcp 65001 >nul
set "PKG=%~dp0"

echo.
echo   Jasna  Portable  -  First run setup
echo   ==================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%PKG%first_run_setup.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
