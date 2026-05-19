"""
SOC tools exposed to the SENTINEL-Ω agent.

In production, these would integrate with SIEM (Splunk), EDR (CrowdStrike),
IdP (Okta), and threat-intel feeds. For the demo and benchmark these are
mocked with realistic responses.
"""

import asyncio
import random
from typing import Any


TOOL_SCHEMAS = [
    {
        "name": "lookup_ip_reputation",
        "description": "Query threat-intel feeds for an IP address.",
        "input_schema": {
            "type": "object",
            "properties": {"ip": {"type": "string"}},
            "required": ["ip"],
        },
    },
    {
        "name": "search_logs",
        "description": "Search SIEM logs over the last 24h.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "minutes": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_user_context",
        "description": "Retrieve identity context: role, MFA, recent logins, travel calendar.",
        "input_schema": {
            "type": "object",
            "properties": {"username": {"type": "string"}},
            "required": ["username"],
        },
    },
    {
        "name": "check_scheduled_jobs",
        "description": "Verify whether activity matches a scheduled job (backup, CI, etc).",
        "input_schema": {
            "type": "object",
            "properties": {"username": {"type": "string"}, "fingerprint": {"type": "string"}},
            "required": ["username"],
        },
    },
    {
        "name": "isolate_host",
        "description": "Network-isolate a host via EDR. Containment action.",
        "input_schema": {
            "type": "object",
            "properties": {"hostname": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["hostname", "reason"],
        },
    },
    {
        "name": "revoke_user_sessions",
        "description": "Revoke all active SSO sessions/tokens for a user.",
        "input_schema": {
            "type": "object",
            "properties": {"username": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["username", "reason"],
        },
    },
    {
        "name": "block_ip",
        "description": "Add an IP to the perimeter firewall blocklist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ip": {"type": "string"},
                "duration_hours": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["ip", "reason"],
        },
    },
    {
        "name": "escalate_to_analyst",
        "description": "Page a human Tier-2 analyst.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "urgency": {"type": "string", "enum": ["low","medium","high","critical"]},
            },
            "required": ["summary", "urgency"],
        },
    },
    {
        "name": "finalize_incident",
        "description": "Close incident with a final verdict + calibrated severity set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string",
                            "enum": ["benign","false_positive","confirmed_threat","inconclusive"]},
                "severity": {"type": "string",
                             "enum": ["info","low","medium","high","critical"]},
                "severity_set": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Conformal prediction set at α=0.1 (≥90% coverage).",
                },
                "confidence": {"type": "number"},
                "summary": {"type": "string"},
                "actions_taken": {"type": "array", "items": {"type": "string"}},
                "escalate": {"type": "boolean"},
            },
            "required": ["verdict","severity","summary","actions_taken","escalate"],
        },
    },
]


_KNOWN_BAD_IPS = {
    "185.220.101.45": {"score": 95, "tags": ["tor_exit","scanning","brute_force"], "country": "DE"},
    "45.155.205.233": {"score": 89, "tags": ["c2_server","emotet"], "country": "RU"},
    "194.165.16.77":  {"score": 78, "tags": ["proxy","credential_stuffing"], "country": "NL"},
    "91.219.236.222": {"score": 82, "tags": ["malware_dropper"], "country": "RU"},
    "146.70.45.18":   {"score": 71, "tags": ["vpn_anonymiser"], "country": "CH"},
}

_USER_DB = {
    "alice.chen":      {"role":"Senior Engineer","mfa":True,
                        "recent_locations":["Seattle","Seattle","Seattle"],
                        "travel_calendared": False},
    "bob.kowalski":    {"role":"Sales Director","mfa":True,
                        "recent_locations":["NYC","NYC","Madrid"],
                        "travel_calendared": True,
                        "calendar_event":"Q3 European Customer Tour"},
    "carol.evans":     {"role":"HR Manager","mfa":True,
                        "recent_locations":["Austin","Austin","Austin"],
                        "departure_date":"2026-05-15"},
    "raj.patel":       {"role":"Security Engineer","mfa":True,
                        "recent_locations":["Seattle"], "pentest_authorised": True},
    "svc-backup-prod": {"role":"Service Account","mfa":False,
                        "scheduled_jobs":["nightly-backup-022","weekly-archive-007"]},
}


async def _delay():
    await asyncio.sleep(random.uniform(0.2, 0.6))


async def execute_tool(name: str, args: dict) -> Any:
    await _delay()

    if name == "lookup_ip_reputation":
        ip = args["ip"]
        if ip in _KNOWN_BAD_IPS:
            return {"ip": ip, "reputation": "malicious", **_KNOWN_BAD_IPS[ip]}
        return {"ip": ip, "reputation": "clean",
                "score": random.randint(0,15), "tags": [],
                "country": random.choice(["US","CA","GB","DE"])}

    if name == "search_logs":
        return {"query": args["query"], "count": random.randint(2,8),
                "events": [{"ts":"2026-05-13T11:00:00Z",
                            "event":random.choice(["auth.login.success","auth.login.failure"]),
                            "detail":f"matched: {args['query']}"} for _ in range(3)]}

    if name == "get_user_context":
        u = args["username"]
        if u in _USER_DB:
            return {"username": u, "found": True, **_USER_DB[u]}
        return {"username": u, "found": False}

    if name == "check_scheduled_jobs":
        u = args["username"]
        info = _USER_DB.get(u, {})
        scheduled = info.get("scheduled_jobs", [])
        return {"username": u, "scheduled_jobs": scheduled,
                "is_currently_scheduled": len(scheduled) > 0}

    if name in ("isolate_host","revoke_user_sessions","block_ip","escalate_to_analyst"):
        return {"status": "executed", **args,
                "ticket": f"INC-{random.randint(10000,99999)}"}

    if name == "finalize_incident":
        return {"status": "closed", **args}

    return {"error": f"Unknown tool: {name}"}
