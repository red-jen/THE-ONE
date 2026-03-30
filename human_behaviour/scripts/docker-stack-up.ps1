# Start the full stack (resolves common "container name already in use" on Windows).
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $here

$names = @(
    "mlflow-server",
    "hb-prometheus",
    "hb-grafana",
    "human-behaviour-api",
    "human-behaviour-frontend",
    "hb-postgres"
)
foreach ($n in $names) {
    docker rm -f $n 2>$null | Out-Null
}

docker compose up -d --build
docker compose ps
