"""Home Assistant REST API integration.
Set JARVIS_HA_URL (e.g. http://homeassistant.local:8123)
    JARVIS_HA_TOKEN (long-lived access token from HA profile)
"""
import os
import requests


def _api():
    url = os.getenv("JARVIS_HA_URL", "").rstrip("/")
    token = os.getenv("JARVIS_HA_TOKEN", "")
    if not url or not token:
        return None, "home assistant not configured"
    return {"base": url, "headers": {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }}, None


def ha_list_entities(domain: str = None) -> str:
    cfg, e = _api()
    if e:
        return e
    try:
        r = requests.get(f"{cfg['base']}/api/states", headers=cfg["headers"],
                         timeout=10)
        r.raise_for_status()
        states = r.json()
        if domain:
            states = [s for s in states if s["entity_id"].startswith(domain + ".")]
        return "\n".join(f"- {s['entity_id']}: {s['state']}" for s in states[:100])
    except Exception as e:
        return f"ERROR: {e}"


def ha_state(entity_id: str) -> str:
    cfg, e = _api()
    if e:
        return e
    try:
        r = requests.get(f"{cfg['base']}/api/states/{entity_id}",
                         headers=cfg["headers"], timeout=10)
        r.raise_for_status()
        s = r.json()
        attrs = s.get("attributes", {})
        return f"{s['entity_id']}: {s['state']}\n  friendly_name: {attrs.get('friendly_name','')}\n  attrs: {attrs}"
    except Exception as e:
        return f"ERROR: {e}"


def ha_call(service: str, entity_id: str = None,
            data: dict = None) -> str:
    """Call a HA service. e.g. service='light.turn_on', entity_id='light.kitchen'"""
    cfg, e = _api()
    if e:
        return e
    domain, _, action = service.partition(".")
    if not action:
        return "service must be in form 'domain.action'"
    payload = data or {}
    if entity_id:
        payload["entity_id"] = entity_id
    try:
        r = requests.post(f"{cfg['base']}/api/services/{domain}/{action}",
                          headers=cfg["headers"], json=payload, timeout=10)
        r.raise_for_status()
        return f"CALLED {service} on {entity_id or payload}"
    except Exception as e:
        return f"ERROR: {e}"


def ha_search(query: str) -> str:
    cfg, e = _api()
    if e:
        return e
    try:
        r = requests.get(f"{cfg['base']}/api/states", headers=cfg["headers"],
                         timeout=10)
        r.raise_for_status()
        q = query.lower()
        states = [s for s in r.json()
                  if q in s["entity_id"] or
                  q in (s.get("attributes", {}).get("friendly_name", "") or "").lower()]
        return "\n".join(f"- {s['entity_id']}: {s['state']}" for s in states[:30])
    except Exception as e:
        return f"ERROR: {e}"