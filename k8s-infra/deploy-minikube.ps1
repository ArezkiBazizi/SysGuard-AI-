# SysGuard-AI - Minikube + Falco + Sidekick + RBAC + DVWA
# Usage: powershell -ExecutionPolicy Bypass -File .\k8s-infra\deploy-minikube.ps1
# Lancer depuis la racine SysGuard-AI- (parent de k8s-infra).
# Encodage recommande: UTF-8 (evite caracteres speciaux dans les chaines ci-dessous = ASCII)

$ErrorActionPreference = "Stop"

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

function Get-HelmExe {
    $cmd = Get-Command helm -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "helm.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $candidates) { throw "helm.exe introuvable. winget install Helm.Helm puis rouvrir le terminal." }
    return $candidates.FullName
}

function Invoke-Kubectl {
    param([string[]]$Arguments)
    $timeoutArg = @("--request-timeout=120s") + $Arguments
    & kubectl @timeoutArg
    if ($LASTEXITCODE -ne 0) { throw "kubectl a echoue: kubectl $($Arguments -join ' ')" }
}

function Test-DockerPs {
    <#
    docker ps peut bloquer longtemps si Docker Desktop demarre ou est bloque.
    On borne l'attente pour eviter un script muet apres "=== 1. Docker ===".
    Affiche stdout/stderr si le code de sortie est non nul.
    #>
    param([int]$TimeoutSec = 45)

    Write-Host "  -> docker ps (max. ${TimeoutSec}s)..." -ForegroundColor Gray
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "docker"
    $psi.Arguments = "ps"
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        try { $p.Kill() } catch {}
        throw "Docker ne repond pas dans ${TimeoutSec}s (docker ps bloque). Ouvrez Docker Desktop, attendez que le moteur soit pret (Engine running), puis relancez ce script."
    }
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    if ($p.ExitCode -ne 0) {
        Write-Host "  Sortie docker ps (code $($p.ExitCode)):" -ForegroundColor Yellow
        if ($stdout.Trim()) { Write-Host $stdout }
        if ($stderr.Trim()) { Write-Host $stderr -ForegroundColor Red }
        return $false
    }
    return $true
}

Write-Host "=== 1. Docker ===" -ForegroundColor Cyan
$dockerOk = Test-DockerPs -TimeoutSec 45
if (-not $dockerOk) {
    Write-Host "`nIndications :" -ForegroundColor Yellow
    Write-Host "  - Si Docker Desktop parle de 'Virtual Machine Platform' : activer la fonctionnalite Windows (admin), redemarrer le PC, puis relancer Docker." -ForegroundColor Gray
    Write-Host "  - Verifier aussi : Docker Desktop demarre, pas de VPN/proxy bloquant, puis 'docker ps' a la main." -ForegroundColor Gray
    throw "Echec de docker ps (voir messages ci-dessus). Corrigez Docker avant de relancer ce script."
}
Write-Host "  -> Docker OK" -ForegroundColor Green

Write-Host "=== 2. Minikube ===" -ForegroundColor Cyan
# Ne pas utiliser "| Out-Null" juste apres minikube: le code de sortie peut etre celui du cmdlet, pas de minikube.
$null = minikube status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Cluster absent ou arrete - lancement: minikube start --driver=docker" -ForegroundColor Yellow
    minikube start --driver=docker
    if ($LASTEXITCODE -ne 0) {
        throw "minikube start a echoue. Verifiez: docker ps, puis relancez ce script."
    }
}
$null = minikube update-context 2>&1

Write-Host "Attente de l'API Kubernetes (jusqu'a ~3 min)..." -ForegroundColor Gray
$apiOk = $false
for ($i = 0; $i -lt 36; $i++) {
    $null = kubectl get nodes --request-timeout=10s 2>&1
    if ($LASTEXITCODE -eq 0) { $apiOk = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $apiOk) {
    throw "API Kubernetes inaccessible (kubeconfig / port refuse). Essayez: minikube delete; minikube start --driver=docker; puis relancez ce script."
}

Write-Host "Attente du noeud Ready..." -ForegroundColor Gray
kubectl wait --request-timeout=120s --for=condition=Ready nodes --all --timeout=300s 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "kubectl wait: echec ou timeout - poursuite avec get nodes..." -ForegroundColor Yellow
}

Invoke-Kubectl @("get", "nodes")

$helm = Get-HelmExe
$values = Join-Path $PSScriptRoot "falco-values.yaml"

Write-Host "`n=== 3. Helm ($helm) ===" -ForegroundColor Cyan
& $helm version

Write-Host "`nhelm repo add / update..." -ForegroundColor Cyan
& $helm repo add falcosecurity https://falcosecurity.github.io/charts 2>$null
& $helm repo update

Write-Host "`nhelm upgrade --install falco..." -ForegroundColor Cyan
& $helm upgrade --install falco falcosecurity/falco -f $values -n falco --create-namespace

Write-Host "`n=== 4. RBAC + DVWA ===" -ForegroundColor Cyan
Invoke-Kubectl @("apply", "-f", (Join-Path $PSScriptRoot "rbac.yaml"))
Invoke-Kubectl @("apply", "-f", (Join-Path $PSScriptRoot "victim-app.yaml"))

Write-Host "`n--- Pods falco ---" -ForegroundColor Green
kubectl get pods -n falco --request-timeout=120s
Write-Host "`n--- Pods default (DVWA) ---" -ForegroundColor Green
kubectl get pods -n default --request-timeout=120s

Write-Host "`nOK. Prochaine etape: image Docker + kubectl apply -f k8s-infra/agent-deployment.yaml" -ForegroundColor Yellow
