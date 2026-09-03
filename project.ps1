param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "validate", "eda-ieee", "eda-paysim", "prepare-ieee", "prepare-paysim", "train-ieee", "train-paysim", "train-paysim-strict", "benchmark-ieee", "benchmark-paysim", "results", "results-strict")]
    [string]$Action = "help",
    [int]$Rows = 0,
    [int]$Epochs = 10,
    [int]$Seed = 42
)

$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "No se encontro .venv. Ejecuta la instalacion indicada en README.md."
}

Push-Location $ProjectRoot
try {
    switch ($Action) {
        "validate" {
            & $Python -m pytest -q -p no:cacheprovider
        }
        "eda-ieee" {
            if ($Rows -le 0) { $Rows = 100000 }
            & $Python eda.py --experiment ieee --sample-mode uniform --max-rows $Rows --seed $Seed
        }
        "eda-paysim" {
            if ($Rows -le 0) { $Rows = 100000 }
            & $Python eda.py --experiment paysim --sample-mode uniform --max-rows $Rows --seed $Seed
        }
        "prepare-ieee" {
            $Arguments = @("run_experiment.py", "--config", "configs/experiment_1_ieee.yaml", "--stage", "prepare")
            if ($Rows -gt 0) { $Arguments += @("--max-rows", $Rows) }
            & $Python @Arguments
        }
        "prepare-paysim" {
            $Arguments = @("run_experiment.py", "--config", "configs/experiment_2_paysim.yaml", "--stage", "prepare", "--sample-mode", "window")
            if ($Rows -gt 0) { $Arguments += @("--max-rows", $Rows) }
            & $Python @Arguments
        }
        "train-ieee" {
            if ($Rows -le 0) { $Rows = 100000 }
            & $Python run_experiment.py --config configs/experiment_1_ieee.yaml --stage all --max-rows $Rows --epochs $Epochs --seed $Seed
        }
        "train-paysim" {
            if ($Rows -le 0) { $Rows = 300000 }
            & $Python run_experiment.py --config configs/experiment_2_paysim.yaml --stage all --sample-mode window --max-rows $Rows --epochs $Epochs --seed $Seed
        }
        "train-paysim-strict" {
            if ($Rows -le 0) { $Rows = 300000 }
            & $Python run_experiment.py --config configs/experiment_2_paysim_strict.yaml --stage all --sample-mode window --max-rows $Rows --epochs $Epochs --seed $Seed
        }
        "benchmark-ieee" {
            if ($Rows -le 0) { $Rows = 100000 }
            foreach ($CurrentSeed in @(42, 52, 62)) {
                Write-Host "`nIEEE-CIS: semilla $CurrentSeed"
                & $Python run_experiment.py --config configs/experiment_1_ieee.yaml --stage all --max-rows $Rows --epochs $Epochs --seed $CurrentSeed
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            & $Python show_results.py --experiment ieee --details --aggregate-seeds
        }
        "benchmark-paysim" {
            if ($Rows -le 0) { $Rows = 300000 }
            foreach ($CurrentSeed in @(42, 52, 62)) {
                Write-Host "`nPaySim: semilla $CurrentSeed"
                & $Python run_experiment.py --config configs/experiment_2_paysim.yaml --stage all --sample-mode window --max-rows $Rows --epochs $Epochs --seed $CurrentSeed
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            }
            & $Python show_results.py --experiment paysim --details --aggregate-seeds
        }
        "results" {
            & $Python show_results.py --experiment all --details
        }
        "results-strict" {
            & $Python show_results.py --experiment paysim-strict --details
        }
        default {
            Write-Host "Comandos disponibles:"
            Write-Host "  .\project.ps1 validate"
            Write-Host "  .\project.ps1 eda-ieee -Rows 100000"
            Write-Host "  .\project.ps1 eda-paysim -Rows 100000"
            Write-Host "  .\project.ps1 prepare-ieee -Rows 100000"
            Write-Host "  .\project.ps1 prepare-paysim -Rows 300000"
            Write-Host "  .\project.ps1 train-ieee -Rows 100000 -Epochs 10 -Seed 42"
            Write-Host "  .\project.ps1 train-paysim -Rows 300000 -Epochs 10 -Seed 42"
            Write-Host "  .\project.ps1 train-paysim-strict -Rows 300000 -Epochs 10 -Seed 42"
            Write-Host "  .\project.ps1 benchmark-ieee -Rows 100000 -Epochs 30"
            Write-Host "  .\project.ps1 benchmark-paysim -Rows 300000 -Epochs 30"
            Write-Host "  .\project.ps1 results"
            Write-Host "  .\project.ps1 results-strict"
        }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
