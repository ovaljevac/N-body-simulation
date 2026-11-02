param(
    [string]$ToolsBin = '',
    [switch]$Native,
    [switch]$DebugBuild,
    [switch]$DoublePrecision
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
$buildName = if ($DoublePrecision) { 'build-double' } else { 'build' }
if ($DebugBuild) { $buildName += '-debug' }
$build = Join-Path $repoRoot $buildName
$nativeFlag = if ($Native) { 'ON' } else { 'OFF' }
$doubleFlag = if ($DoublePrecision) { 'ON' } else { 'OFF' }
# FP32 i FP64 koriste isti čitljiv kod i odvojene build foldere.
# Ostatak sistema ostaje nepromijenjen; PATH se dopunjava samo u ovom procesu.
& $cmakeExe -S $repoRoot -B $build -G 'MinGW Makefiles' "-DCMAKE_BUILD_TYPE=$config" "-DNBODY_NATIVE=$nativeFlag" "-DNBODY_DOUBLE=$doubleFlag"
if ($LASTEXITCODE) { throw 'CMake configuration failed' }
& $cmakeExe --build $build --parallel 4
if ($LASTEXITCODE) { throw 'Compilation failed' }
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir $build --output-on-failure
if ($LASTEXITCODE) { throw 'Tests failed' }
