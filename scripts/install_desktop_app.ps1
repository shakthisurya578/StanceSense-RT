# Adds (or removes) the StanceSense-RT desktop app shortcuts on this PC.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_desktop_app.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_desktop_app.ps1 -Uninstall
#
# Creates "StanceSense-RT" on the Desktop and in the Start menu. Each one runs
# scripts\desktop_app.py with the project's .venv pythonw.exe (no console window).
# Nothing is copied or installed elsewhere: the shortcuts point at this folder, so
# move the project and you need to run this again. -Uninstall removes the shortcuts
# and the app's browser profile and logs under %LOCALAPPDATA%\StanceSense-RT.
param([switch]$Uninstall)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
$launcher = Join-Path $root "scripts\desktop_app.py"
$icon = Join-Path $root "assets\stancesense.ico"
$places = @(
    [Environment]::GetFolderPath("Desktop"),
    [Environment]::GetFolderPath("Programs")      # Start menu > All apps
)

if ($Uninstall) {
    foreach ($dir in $places) {
        $lnk = Join-Path $dir "StanceSense-RT.lnk"
        if (Test-Path $lnk) { Remove-Item $lnk; "removed $lnk" }
    }
    $data = Join-Path $env:LOCALAPPDATA "StanceSense-RT"
    if (Test-Path $data) { Remove-Item $data -Recurse -Force; "removed $data" }
    return
}

if (-not (Test-Path $pythonw)) {
    throw "No virtual environment at $pythonw. Create .venv and install requirements first (see README)."
}
if (-not (Test-Path $icon)) {
    & (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $root "scripts\make_app_icon.py") --out-dir (Join-Path $root "assets")
}

$shell = New-Object -ComObject WScript.Shell
foreach ($dir in $places) {
    $lnk = Join-Path $dir "StanceSense-RT.lnk"
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = $pythonw
    $s.Arguments = "`"$launcher`""
    $s.WorkingDirectory = $root
    $s.IconLocation = "$icon,0"
    $s.Description = "StanceSense-RT: hip-rotation profile, stance advice and squat check"
    $s.Save()
    "created $lnk"
}
