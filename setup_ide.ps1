# Generate compile_commands.json for IDE support
Write-Host "Setting up IDE configuration..." -ForegroundColor Cyan

# 1. Find CMake
$cmakeCmd = "cmake"
if (Get-Command cmake -ErrorAction SilentlyContinue) {
    # cmake in path
} elseif (Test-Path "C:\Program Files\CMake\bin\cmake.exe") {
    $cmakeCmd = "C:\Program Files\CMake\bin\cmake.exe"
} elseif (Test-Path "C:\Program Files (x86)\CMake\bin\cmake.exe") {
    $cmakeCmd = "C:\Program Files (x86)\CMake\bin\cmake.exe"
} else {
    Write-Host "ERROR: CMake not found. Please install CMake." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# 2. Check for C++ Compiler and Setup Environment
if (-not (Get-Command "cl" -ErrorAction SilentlyContinue)) {
    Write-Host "C++ compiler (cl.exe) not found in PATH. Searching for Visual Studio..." -ForegroundColor Yellow
    
    $vsYears = @("2026", "2025", "2022")
    $vsEditions = @("Enterprise", "Professional", "Community", "Preview", "BuildTools")
    $vcvarsPath = $null
    
    foreach ($year in $vsYears) {
        foreach ($edition in $vsEditions) {
            $path = "C:\Program Files\Microsoft Visual Studio\$year\$edition\VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $path) {
                $vcvarsPath = $path
                Write-Host "Found Visual Studio $year $edition" -ForegroundColor Green
                break
            }
        }
        if ($vcvarsPath) { break }
    }
    
    if ($vcvarsPath) {
        # We found vcvars, but we can't easily source it into PowerShell.
        # Strategy: Run cmake via cmd.exe /c calling vcvars first.
        
        if (-not (Test-Path "build")) {
            New-Item -ItemType Directory -Path "build" | Out-Null
        }
        
        Write-Host "Running CMake via cmd.exe to load VS environment..." -ForegroundColor Cyan
        
        $cmdArgs = "/c `"`"$vcvarsPath`" && `"$cmakeCmd`" -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`""
        Start-Process "cmd.exe" -ArgumentList $cmdArgs -Wait -NoNewWindow
        
        # Check result by looking for file
        if (Test-Path "build\compile_commands.json") {
            Copy-Item "build\compile_commands.json" "compile_commands.json" -Force
            Write-Host ""
            Write-Host "SUCCESS: compile_commands.json generated." -ForegroundColor Green
            Write-Host "Please restart your IDE or reload the window." -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "ERROR: CMake configuration failed or compile_commands.json not generated." -ForegroundColor Red
            Write-Host "Check output above for details."
        }
        
        Read-Host "Press Enter to exit"
        exit 0
    } else {
        Write-Host "WARNING: Could not find Visual Studio installation." -ForegroundColor Yellow
    }
}

# 3. Standard Flow (if cl is in path or we fell through)
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Push-Location build

Write-Host "Running CMake..." -ForegroundColor Cyan
& $cmakeCmd .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: CMake configuration failed." -ForegroundColor Red
    Pop-Location
    Read-Host "Press Enter to exit"
    exit 1
}

if (Test-Path "compile_commands.json") {
    Copy-Item "compile_commands.json" "..\compile_commands.json" -Force
    Write-Host ""
    Write-Host "SUCCESS: compile_commands.json generated." -ForegroundColor Green
    Write-Host "Please restart your IDE or reload the window." -ForegroundColor Green
} else {
    Write-Host "WARNING: compile_commands.json was not generated." -ForegroundColor Yellow
}

Pop-Location
Read-Host "Press Enter to exit"
