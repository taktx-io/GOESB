# GOESB runner installer for Windows -- downloads the matching standalone
# PyInstaller binary from the latest GitHub release and installs it as
# goesb.exe on PATH. No Python required.
#
# This is one of several ways to install the runner -- see
# https://goesb.com/docs/how-to for pipx/pip (needed on Windows-on-ARM,
# which this script doesn't cover) and building from source.
#
# Usage (from a fresh PowerShell -- piped scripts can't take -Engine/-Dir
# flags directly, so use environment variables instead):
#   irm https://raw.githubusercontent.com/taktx-io/GOESB/main/scripts/install.ps1 | iex
#   $env:GOESB_ENGINE = "vosk"; irm https://raw.githubusercontent.com/taktx-io/GOESB/main/scripts/install.ps1 | iex
#
# Downloaded and run directly, -Engine/-InstallDir work normally:
#   .\install.ps1 -Engine vosk -InstallDir C:\tools\goesb

param(
    [string]$Engine = $(if ($env:GOESB_ENGINE) { $env:GOESB_ENGINE } else { "faster-whisper" }),
    [string]$InstallDir = $(if ($env:GOESB_INSTALL_DIR) { $env:GOESB_INSTALL_DIR } else { "$env:LOCALAPPDATA\goesb\bin" })
)

$ErrorActionPreference = "Stop"

$Repo = "taktx-io/GOESB"
$DocsUrl = "https://goesb.com/docs/how-to"

if ($Engine -notin @("faster-whisper", "vosk", "whisper-cpp")) {
    Write-Error "unknown engine '$Engine' (expected faster-whisper, vosk, or whisper-cpp)"
    exit 1
}

$Arch = $env:PROCESSOR_ARCHITECTURE
if ($Arch -ne "AMD64") {
    Write-Error "no prebuilt Windows binary for architecture '$Arch' (only x64/AMD64 is built). Install via pipx/pip instead -- see $DocsUrl"
    exit 1
}

$AssetName = "goesb-$Engine-windows-x64.exe"
$Url = "https://github.com/$Repo/releases/latest/download/$AssetName"

Write-Host "Downloading $AssetName ..."

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$Dest = Join-Path $InstallDir "goesb-$Engine.exe"

try {
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
} catch {
    Write-Error "failed to download $Url -- check https://github.com/$Repo/releases/latest for available assets."
    exit 1
}

# Keep each engine's binary under its own name, and also drop a plain
# goesb.exe copy pointing at whichever was installed most recently --
# Windows has no reliable equivalent of a Unix symlink without extra
# privileges, so this copies rather than links.
$PlainDest = Join-Path $InstallDir "goesb.exe"
Copy-Item -Path $Dest -Destination $PlainDest -Force

Write-Host "Installed goesb-$Engine to $Dest (copied as $PlainDest)"

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host ""
    Write-Host "Added $InstallDir to your user PATH. Restart your terminal for it to take effect."
}

Write-Host ""
Write-Host "Run 'goesb --help' to get started (after restarting your terminal, if PATH was just updated)."
