param(
    [string]$ToolsBin = '',
    [switch]$Native,
    [switch]$DebugBuild
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspaceTools = Join-Path $PSScriptRoot '../../../../tmp/toolchains'
if (!$ToolsBin -and (Test-Path (Join-Path $workspaceTools 'w64devkit/bin/g++.exe'))) {
    $ToolsBin = (Resolve-Path (Join-Path $workspaceTools 'w64devkit/bin')).Path
}
if ($ToolsBin) { $env:PATH = $ToolsBin + ';' + $env:PATH }
$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) { $cmakeExe = $cmakeCommand.Source }
else {
    $cmakeExe = Join-Path $workspaceTools 'cmake/cmake/data/bin/cmake.exe'
    if (!(Test-Path -LiteralPath $cmakeExe)) { throw 'Install CMake and GCC/OpenMP or provide -ToolsBin.' }
    $cmakeExe = (Resolve-Path -LiteralPath $cmakeExe).Path
}
$config = if ($DebugBuild) { 'Debug' } else { 'Release' }
$build = Join-Path $repoRoot $(if ($DebugBuild) { 'build-debug' } else { 'build' })
$nativeFlag = if ($Native) { 'ON' } else { 'OFF' }
& $cmakeExe -S $repoRoot -B $build -G 'MinGW Makefiles' "-DCMAKE_BUILD_TYPE=$config" "-DNBODY_NATIVE=$nativeFlag"
if ($LASTEXITCODE) { throw 'CMake configuration failed' }
& $cmakeExe --build $build --parallel 4
if ($LASTEXITCODE) { throw 'Compilation failed' }
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir $build --output-on-failure
if ($LASTEXITCODE) { throw 'Tests failed' }
