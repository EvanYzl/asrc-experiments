param()
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    New-Item -ItemType Directory -Path (Join-Path $projectDirectory 'build') -Force | Out-Null
    $compilerCommand = Get-Command latexmk -ErrorAction Stop
    & $compilerCommand.Source -norc -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $LASTEXITCODE" }
    Copy-Item -LiteralPath (Join-Path $projectDirectory 'build\main.pdf') -Destination (Join-Path $projectDirectory 'KBS_Main_Text_Tables.pdf') -Force
    Write-Host 'Compiled: KBS_Main_Text_Tables.pdf'
} finally { Pop-Location }

