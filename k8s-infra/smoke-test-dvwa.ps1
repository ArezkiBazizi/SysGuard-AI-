# Test bout-en-bout : shell dans DVWA -> Falco -> Sidekick -> agent SysGuard-AI
# Prerequis : Falco Running, image sysguard-ai:latest visible par Minikube, agent deploye

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Agent Running ? ===" -ForegroundColor Cyan
kubectl get pods -n default -l app=sysguard-agent
kubectl wait --for=condition=ready pod -l app=sysguard-agent -n default --timeout=120s

Write-Host "`n=== 2. healthz ===" -ForegroundColor Cyan
$pod = kubectl get pod -n default -l app=sysguard-agent -o jsonpath='{.items[0].metadata.name}'
kubectl exec -n default $pod -- curl -s http://127.0.0.1:8000/healthz | Write-Host

Write-Host "`n=== 3. Shell interactif dans DVWA (declenche la regle 'Terminal shell in container') ===" -ForegroundColor Cyan
$dvw = kubectl get pod -n default -l app=dvwa -o jsonpath='{.items[0].metadata.name}'
Write-Host "Pod DVWA: $dvw"
kubectl exec -n default $dvw -c dvwa -- sh -c "bash -c 'echo SysGuard smoke-test shell dans DVWA'"

Write-Host "`nAttendre ~15-30s (Sidekick + webhook + fenetre tumbling 10s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 25

Write-Host "`n=== 4. Logs agent (dernieres lignes) ===" -ForegroundColor Cyan
kubectl logs -n default $pod -c agent --tail=40

Write-Host "`n=== 5. Incidents (via port-forward ou exec curl) ===" -ForegroundColor Cyan
kubectl exec -n default $pod -- curl -s http://127.0.0.1:8000/incidents?limit=5 | Write-Host

Write-Host "`n=== 6. Logs Falcosidekick (optionnel) ===" -ForegroundColor Cyan
$pods = kubectl get pods -n falco -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | Where-Object { $_ -match 'sidekick' }
foreach ($p in $pods) {
    if ($p) {
        Write-Host "--- $p ---"
        kubectl logs -n falco $p --tail=25
    }
}

Write-Host "`nOK. Verdict 'anomaly' / ANOMALIE dans les logs = pipeline actif." -ForegroundColor Green
