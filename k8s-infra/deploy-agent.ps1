# Build image sysguard-ai + apply agent-deployment.yaml
# Depuis la racine SysGuard-AI- (parent de k8s-infra)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

Set-Location $proj

Write-Host "=== Docker build (PyTorch CPU dans Dockerfile) ===" -ForegroundColor Cyan
Write-Host "  Premiere fois : souvent 5 a 15 min (torch CPU ~150 Mo + deps). La sortie peut rester vide" -ForegroundColor Gray
Write-Host "  plusieurs minutes pendant les telechargements - ne fermez pas le terminal.`n" -ForegroundColor Gray

$env:DOCKER_BUILDKIT = "1"
docker build --progress=plain -f agent/Dockerfile -t sysguard-ai:latest .

if ($LASTEXITCODE -ne 0) { throw "docker build a echoue" }

Write-Host "`n=== Minikube : charger image dans le cluster si besoin ===" -ForegroundColor Cyan
minikube image load sysguard-ai:latest 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "minikube image load non necessaire ou echoue -- si le cluster utilise le meme Docker, continuez." -ForegroundColor Yellow
}

Write-Host "`n=== kubectl apply agent ===" -ForegroundColor Cyan
kubectl apply -f (Join-Path $PSScriptRoot "agent-deployment.yaml")

kubectl rollout restart deployment/sysguard-agent -n default 2>$null
kubectl rollout status deployment/sysguard-agent -n default --timeout=180s

kubectl get pods -n default -l app=sysguard-agent
Write-Host "`nEnsuite : powershell -ExecutionPolicy Bypass -File .\k8s-infra\smoke-test-dvwa.ps1" -ForegroundColor Green
