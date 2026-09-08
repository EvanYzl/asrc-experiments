param()
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    New-Item -ItemType Directory -Path (Join-Path $projectDirectory 'build') -Force | Out-Null
    $compilerCommand = Get-Command latexmk -ErrorAction Stop
    & $compilerCommand.Source -norc -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed with exit code $LASTEXITCODE." }
    $compiledPdf = Join-Path $projectDirectory 'build\main.pdf'
    if (-not (Test-Path -LiteralPath $compiledPdf)) { throw 'The compiled PDF was not produced.' }
    Copy-Item -LiteralPath $compiledPdf -Destination (Join-Path $projectDirectory 'QURA_Cert_Experiment_Tables.pdf') -Force
    Write-Host 'Compiled: QURA_Cert_Experiment_Tables.pdf'
} finally {
    Pop-Location
}

