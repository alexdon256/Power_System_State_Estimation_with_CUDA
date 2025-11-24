@echo off
setlocal enabledelayedexpansion

echo Setting up IDE configuration...

:: 1. Find CMake
where cmake >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "CMAKE_CMD=cmake"
) else (
    if exist "C:\Program Files\CMake\bin\cmake.exe" (
        set "CMAKE_CMD=C:\Program Files\CMake\bin\cmake.exe"
    ) else if exist "C:\Program Files (x86)\CMake\bin\cmake.exe" (
        set "CMAKE_CMD=C:\Program Files (x86)\CMake\bin\cmake.exe"
    ) else (
        echo ERROR: CMake not found. Please install CMake and add it to PATH.
        pause
        exit /b 1
    )
)

:: 2. Check for C++ Compiler (cl.exe)
where cl >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo C++ compiler not found in PATH. Searching for Visual Studio...
    
    set "FOUND_VS="
    
    :: Define search order: Newer versions first, Community prioritized (most common for individual developers)
    set "VS_YEARS=2026 2025 2022 2019"
    set "VS_EDITIONS=Community Enterprise Professional Preview BuildTools"
    
    for %%y in (!VS_YEARS!) do (
        for %%e in (!VS_EDITIONS!) do (
            if not defined FOUND_VS (
                set "VS_PATH=C:\Program Files\Microsoft Visual Studio\%%y\%%e\VC\Auxiliary\Build\vcvars64.bat"
                if exist "!VS_PATH!" (
                    echo Found Visual Studio %%y %%e - Initializing compiler environment...
                    call "!VS_PATH!" >nul
                    if !ERRORLEVEL! EQU 0 (
                        set "FOUND_VS=1"
                    )
                )
            )
        )
    )
    
    if not defined FOUND_VS (
        echo WARNING: Could not find Visual Studio installation.
        echo Searched for: Visual Studio 2026, 2025, 2022, 2019
        echo Editions: Community, Enterprise, Professional, Preview, BuildTools
        echo CMake might fail if no C++ compiler is available.
        echo.
        echo Please ensure Visual Studio is installed with C++ workload, or run this script
        echo from a Visual Studio Developer Command Prompt.
    )
)

:: 3. Create/clean build directory
if exist build (
    echo Cleaning previous build configuration...
    rd /s /q build
)
mkdir build
cd build

:: 4. Run CMake
echo Running CMake...
echo Letting CMake auto-detect Visual Studio generator from environment...
:: Let CMake auto-detect the generator from the environment set by vcvars64.bat
:: This avoids path mismatches - CMake will detect the correct VS installation
:: based on environment variables (VCINSTALLDIR, etc.) set by vcvars64.bat
call "%CMAKE_CMD%" .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed.
    echo Possible reasons:
    echo  1. CUDA Toolkit is not installed or not found.
    echo  2. Visual Studio compiler is not compatible.
    echo  3. CMake version is too old for this Visual Studio version.
    echo  4. For MSVC 19.50+ (VS 2026, e.g., 19.50.35718.0): Ensure CUDA 12.1+ is installed.
    echo     The -allow-unsupported-compiler flag will be automatically added.
    echo.
    cd ..
    pause
    exit /b 1
)

:: 5. Copy compile_commands.json
if exist compile_commands.json (
    copy compile_commands.json ..\compile_commands.json >nul
    echo.
    echo SUCCESS: compile_commands.json generated.
    echo Configuration uses: Visual Studio compiler and CUDA.
    echo Please restart your IDE or reload the window.
) else (
    echo WARNING: compile_commands.json was not generated.
)

cd ..
pause
