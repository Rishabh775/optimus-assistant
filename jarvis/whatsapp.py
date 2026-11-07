"""WhatsApp channel via the local bridge (whatsapp/).

Outbound: notifications and agent replies are POSTed to the bridge, which
sends them through the linked WhatsApp device (self-chat by default).
Inbound: the bridge calls /api/whatsapp/inbound on the Soundwave server.

Unofficial WhatsApp clients violate WhatsApp's ToS; this is a personal,
low-volume setup.
"""
import json
import urllib.request

from . import config
from .log import log as _mk_log

log = _mk_log("whatsapp")


def configured() -> bool:
    return (config.get_bool("JARVIS_WHATSAPP_ENABLED")
            and bool(config.get("JARVIS_WHATSAPP_BRIDGE_URL"))
            and bool(config.get("JARVIS_WHATSAPP_BRIDGE_TOKEN")))


def send(text: str, to: str = "") -> bool:
    """Send a WhatsApp message via the bridge. True when accepted."""
    if not configured():
        return False
    url = config.get("JARVIS_WHATSAPP_BRIDGE_URL").rstrip("/") + "/send"
    payload = json.dumps({
        "text": str(text or "")[:6000],
        "to": to or config.get("JARVIS_WHATSAPP_TO"),
    }).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " +
                             config.get("JARVIS_WHATSAPP_BRIDGE_TOKEN"),
        })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
        return True
    except Exception as e:
        log.warning("whatsapp send failed: %s", e)
        return False


def notify(title: str, message: str) -> bool:
    """Notification-style message (WhatsApp bold title)."""
    return send(f"*{title}*\n{message}")
