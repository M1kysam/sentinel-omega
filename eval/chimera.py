"""
CHIMERA — A synthetic benchmark of multi-stage attack scenarios.

Each scenario is a tuple (alert_stream, ground_truth):
  alert_stream:  list of alerts ordered in time
  ground_truth:  {
    'attack_graph': DAG of MITRE ATT&CK tactics involved,
    'true_severity': one of info|low|medium|high|critical,
    'is_threat': bool,
    'optimal_containment': list of recommended actions,
    'is_false_positive': bool,
  }

The benchmark spans 14 MITRE ATT&CK tactics and 6 archetype attack chains:
  1. CREDENTIAL_STUFFING_TO_LATERAL    (Initial Access → Credential Access → Lateral Movement)
  2. PHISH_TO_C2_TO_EXFIL              (Initial Access → C2 → Exfiltration)
  3. SUPPLY_CHAIN_TO_PERSISTENCE       (Initial Access → Execution → Persistence)
  4. INSIDER_DATA_THEFT                (Valid Accounts → Collection → Exfiltration)
  5. RANSOMWARE_FULL_CHAIN             (full kill chain → Impact)
  6. SERVICE_ACCOUNT_ABUSE             (Cloud-specific privilege escalation)

We also generate FALSE_POSITIVE scenarios to test conservative triage —
benign user behaviour that triggers naive rules.
"""

from __future__ import annotations
import json
import random
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path


ARCHETYPES = [
    "CREDENTIAL_STUFFING_TO_LATERAL",
    "PHISH_TO_C2_TO_EXFIL",
    "SUPPLY_CHAIN_TO_PERSISTENCE",
    "INSIDER_DATA_THEFT",
    "RANSOMWARE_FULL_CHAIN",
    "SERVICE_ACCOUNT_ABUSE",
    "BENIGN_TRAVEL",
    "BENIGN_DEV_TESTING",
    "BENIGN_BACKUP",
]

SEVERITIES = ["info", "low", "medium", "high", "critical"]

# Per-archetype severity / threat priors
ARCHETYPE_PROFILE = {
    "CREDENTIAL_STUFFING_TO_LATERAL": ("high",     True),
    "PHISH_TO_C2_TO_EXFIL":           ("critical", True),
    "SUPPLY_CHAIN_TO_PERSISTENCE":    ("high",     True),
    "INSIDER_DATA_THEFT":             ("high",     True),
    "RANSOMWARE_FULL_CHAIN":          ("critical", True),
    "SERVICE_ACCOUNT_ABUSE":          ("medium",   True),
    "BENIGN_TRAVEL":                  ("info",     False),
    "BENIGN_DEV_TESTING":             ("info",     False),
    "BENIGN_BACKUP":                  ("info",     False),
}

# Realistic IPs / users / hostnames
KNOWN_BAD_IPS = ["185.220.101.45", "45.155.205.233", "194.165.16.77",
                 "91.219.236.222", "146.70.45.18"]
INTERNAL_IPS  = [f"10.{a}.{b}.{c}" for a in range(40,45) for b in range(10,20) for c in range(1,50)]
COUNTRIES_HOSTILE = ["RU", "KP", "IR"]
USERS = ["alice.chen", "bob.kowalski", "carol.evans", "david.wu", "emma.miller",
         "svc-backup-prod", "svc-ci-runner", "raj.patel"]
HOSTS = [f"WS-{dept}-{i:04d}" for dept in ["FIN","ENG","MKT","HR","OPS"] for i in range(1,60)]


@dataclass
class Scenario:
    id: str
    archetype: str
    alerts: list[dict] = field(default_factory=list)
    ground_truth: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-archetype alert generators
# ---------------------------------------------------------------------------

def _gen_credential_stuffing(rng: random.Random, sid: int) -> list[dict]:
    user = rng.choice(USERS)
    ip = rng.choice(KNOWN_BAD_IPS)
    host = rng.choice(HOSTS)
    return [
        {"id": f"{sid}-A1", "title": "Multiple failed logins followed by success",
         "severity": "high", "source": "Okta", "timestamp": "T+0",
         "details": {"username": user, "src_ip": ip,
                     "failed_attempts": rng.randint(30, 200),
                     "user_agent": "curl/7.81", "mfa_used": False}},
        {"id": f"{sid}-A2", "title": "Unusual SSH login from corporate workstation",
         "severity": "medium", "source": "CrowdStrike", "timestamp": "T+8",
         "details": {"username": user, "hostname": host,
                     "dst_hostnames": [rng.choice(HOSTS), rng.choice(HOSTS)]}},
    ]


def _gen_phish_c2_exfil(rng: random.Random, sid: int) -> list[dict]:
    host = rng.choice(HOSTS)
    user = rng.choice(USERS)
    return [
        {"id": f"{sid}-A1", "title": "User opened email attachment from external sender",
         "severity": "low", "source": "Proofpoint", "timestamp": "T+0",
         "details": {"username": user, "attachment": "Q3-financials.docm",
                     "sender_domain": "external-partner.biz"}},
        {"id": f"{sid}-A2", "title": "Beaconing traffic to suspected C2",
         "severity": "critical", "source": "CrowdStrike", "timestamp": "T+45",
         "details": {"hostname": host, "dst_ip": rng.choice(KNOWN_BAD_IPS),
                     "interval_seconds": 60, "bytes_out": rng.randint(8_000, 30_000)}},
        {"id": f"{sid}-A3", "title": "Large outbound transfer",
         "severity": "high", "source": "Zscaler", "timestamp": "T+180",
         "details": {"hostname": host, "bytes_out_mb": rng.randint(500, 4000),
                     "dst": rng.choice(["paste.ee", "transfer.sh", "mega.nz"])}},
    ]


def _gen_supply_chain(rng: random.Random, sid: int) -> list[dict]:
    host = rng.choice(HOSTS)
    return [
        {"id": f"{sid}-A1", "title": "Unsigned binary executed under signed process",
         "severity": "high", "source": "Defender", "timestamp": "T+0",
         "details": {"hostname": host, "parent": "updater.exe",
                     "child": "anomalous_helper.exe"}},
        {"id": f"{sid}-A2", "title": "New persistence registry key created",
         "severity": "medium", "source": "Defender", "timestamp": "T+12",
         "details": {"hostname": host,
                     "key": "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\helper"}},
    ]


def _gen_insider_theft(rng: random.Random, sid: int) -> list[dict]:
    user = rng.choice(["bob.kowalski", "carol.evans", "david.wu"])
    return [
        {"id": f"{sid}-A1", "title": "Bulk file access on departing employee account",
         "severity": "medium", "source": "DLP", "timestamp": "T+0",
         "details": {"username": user, "file_count": rng.randint(800, 4000),
                     "tenure_days": rng.randint(800, 2000),
                     "departure_date_known": True}},
        {"id": f"{sid}-A2", "title": "Personal cloud-storage upload",
         "severity": "high", "source": "Zscaler", "timestamp": "T+25",
         "details": {"username": user, "dst": "drive.google.com",
                     "bytes_out_mb": rng.randint(200, 2000)}},
    ]


def _gen_ransomware(rng: random.Random, sid: int) -> list[dict]:
    host = rng.choice(HOSTS)
    return [
        {"id": f"{sid}-A1", "title": "PowerShell with encoded command",
         "severity": "high", "source": "Defender", "timestamp": "T+0",
         "details": {"hostname": host, "command_length": 4082, "encoded": True}},
        {"id": f"{sid}-A2", "title": "Mass file modification — possible encryption",
         "severity": "critical", "source": "Defender", "timestamp": "T+18",
         "details": {"hostname": host, "files_modified_per_minute": rng.randint(800, 4000),
                     "new_extension": rng.choice([".locked",".crypt",".pay"])}},
        {"id": f"{sid}-A3", "title": "Shadow copy deletion attempt",
         "severity": "critical", "source": "Defender", "timestamp": "T+19",
         "details": {"hostname": host, "command": "vssadmin delete shadows /all /quiet"}},
    ]


def _gen_service_account_abuse(rng: random.Random, sid: int) -> list[dict]:
    return [
        {"id": f"{sid}-A1", "title": "Service account login from unusual region",
         "severity": "medium", "source": "AWS CloudTrail", "timestamp": "T+0",
         "details": {"username": "svc-backup-prod",
                     "src_ip": rng.choice(KNOWN_BAD_IPS),
                     "region": "eu-west-2", "expected_region": "us-east-1"}},
        {"id": f"{sid}-A2", "title": "Service account assumed elevated role",
         "severity": "high", "source": "AWS CloudTrail", "timestamp": "T+5",
         "details": {"username": "svc-backup-prod", "role": "OrgAdmin",
                     "action": "iam:CreateUser"}},
    ]


def _gen_benign_travel(rng: random.Random, sid: int) -> list[dict]:
    user = rng.choice(USERS[:5])
    # Generate IPs that look suspicious but are legitimate travel
    suspicious_ip = rng.choice([
        f"83.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}",  # Spain
        f"185.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}",  # EU range that overlaps tor exits
    ])
    # 30% of benign travel hits a known-bad IP range — VPN/Tor for privacy
    if rng.random() < 0.3:
        suspicious_ip = rng.choice(KNOWN_BAD_IPS)
    return [{"id": f"{sid}-A1", "title": "Login from new country",
             "severity": rng.choice(["medium","high"]), "source": "Okta", "timestamp": "T+0",
             "details": {"username": user, "src_country": rng.choice(["ES","DE","NL"]),
                         "src_ip": suspicious_ip,
                         "mfa_used": True, "travel_calendared": True}}]


def _gen_benign_dev_testing(rng: random.Random, sid: int) -> list[dict]:
    # Security engineer running an internal pentest — looks like Recon
    return [{"id": f"{sid}-A1", "title": "Anomalous process from developer workstation",
             "severity": rng.choice(["high","medium"]), "source": "Defender", "timestamp": "T+0",
             "details": {"username": rng.choice(["alice.chen", "raj.patel"]),
                         "hostname": rng.choice(HOSTS),
                         "process": rng.choice(["nmap.exe","mimikatz.exe","powershell -enc"]),
                         "dst_subnet": "10.0.0.0/16", "user_role": "Security Engineer",
                         "ticket_id": f"PT-2026-{rng.randint(100,999)}"}}]


def _gen_benign_backup(rng: random.Random, sid: int) -> list[dict]:
    # Scheduled backup that triggers DLP and outbound transfer rules
    return [{"id": f"{sid}-A1", "title": "Bulk read on file share",
             "severity": rng.choice(["medium","high"]), "source": "DLP", "timestamp": "T+0",
             "details": {"username": "svc-backup-prod", "file_count": rng.randint(8000,40000),
                         "bytes_out_mb": rng.randint(2000, 10000),
                         "scheduled": True, "schedule_id": f"nightly-backup-{rng.randint(1,99):03d}"}}]


GENERATORS = {
    "CREDENTIAL_STUFFING_TO_LATERAL": _gen_credential_stuffing,
    "PHISH_TO_C2_TO_EXFIL":           _gen_phish_c2_exfil,
    "SUPPLY_CHAIN_TO_PERSISTENCE":    _gen_supply_chain,
    "INSIDER_DATA_THEFT":             _gen_insider_theft,
    "RANSOMWARE_FULL_CHAIN":          _gen_ransomware,
    "SERVICE_ACCOUNT_ABUSE":          _gen_service_account_abuse,
    "BENIGN_TRAVEL":                  _gen_benign_travel,
    "BENIGN_DEV_TESTING":             _gen_benign_dev_testing,
    "BENIGN_BACKUP":                  _gen_benign_backup,
}


# ---------------------------------------------------------------------------
# Benchmark generation
# ---------------------------------------------------------------------------

# Distribution: real attacks are rare in the wild. We use roughly 30% threats /
# 70% benign — generous to threats vs typical SOC base rates of ~1%, but tuned
# so the benchmark is well-balanced for F1 and ECE evaluation.
ARCHETYPE_DIST = {
    "CREDENTIAL_STUFFING_TO_LATERAL": 0.07,
    "PHISH_TO_C2_TO_EXFIL":           0.07,
    "SUPPLY_CHAIN_TO_PERSISTENCE":    0.05,
    "INSIDER_DATA_THEFT":             0.05,
    "RANSOMWARE_FULL_CHAIN":          0.05,
    "SERVICE_ACCOUNT_ABUSE":          0.06,
    "BENIGN_TRAVEL":                  0.25,
    "BENIGN_DEV_TESTING":             0.20,
    "BENIGN_BACKUP":                  0.20,
}


def generate(n: int = 1200, seed: int = 42) -> list[Scenario]:
    """Generate `n` scenarios with the archetype mixture defined above."""
    rng = random.Random(seed)
    archetypes = list(ARCHETYPE_DIST.keys())
    weights = list(ARCHETYPE_DIST.values())
    scenarios = []

    for i in range(n):
        arch = rng.choices(archetypes, weights=weights, k=1)[0]
        sev, is_threat = ARCHETYPE_PROFILE[arch]
        alerts = GENERATORS[arch](rng, sid=f"CHM{i:05d}")
        gt = {
            "archetype": arch,
            "true_severity": sev,
            "is_threat": is_threat,
            "is_false_positive": (not is_threat) and any(
                a["severity"] in ("medium","high","critical") for a in alerts),
            "optimal_containment": _optimal_containment(arch),
        }
        scenarios.append(Scenario(id=f"CHM{i:05d}", archetype=arch,
                                  alerts=alerts, ground_truth=gt))
    return scenarios


def _optimal_containment(archetype: str) -> list[str]:
    return {
        "CREDENTIAL_STUFFING_TO_LATERAL": ["revoke_user_sessions", "block_ip"],
        "PHISH_TO_C2_TO_EXFIL":           ["isolate_host", "block_ip"],
        "SUPPLY_CHAIN_TO_PERSISTENCE":    ["isolate_host", "escalate_to_analyst"],
        "INSIDER_DATA_THEFT":             ["revoke_user_sessions", "escalate_to_analyst"],
        "RANSOMWARE_FULL_CHAIN":          ["isolate_host", "block_ip", "escalate_to_analyst"],
        "SERVICE_ACCOUNT_ABUSE":          ["revoke_user_sessions", "block_ip"],
        "BENIGN_TRAVEL":                  [],
        "BENIGN_DEV_TESTING":             [],
        "BENIGN_BACKUP":                  [],
    }[archetype]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate the CHIMERA benchmark.")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="chimera_dataset.json")
    args = parser.parse_args()

    scenarios = generate(n=args.n, seed=args.seed)
    out = [asdict(s) for s in scenarios]
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} scenarios → {args.out}")

    # Print archetype distribution
    from collections import Counter
    dist = Counter(s.archetype for s in scenarios)
    print("\nArchetype distribution:")
    for k, v in sorted(dist.items()):
        print(f"  {k:38s} {v:5d}  ({100*v/len(scenarios):.1f}%)")


if __name__ == "__main__":
    main()
