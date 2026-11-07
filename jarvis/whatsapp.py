"""WhatsApp Cloud API channel (official, Meta Graph API).

Replaces the unofficial linked-device bridge. Constraints that shape this:

- You can only send a free-form **service message** inside the 24-hour
  customer-service window that opens when the user messages you.
- Outside that window you must send an approved **template**. Set
  JARVIS_WA_TEMPLATE_NOTIFY to a utility template with two body parameters
  (title, message) to enable proactive job/reminder/briefing alerts.
- Since 1 Oct 2026 service messages are billed after a free allowance of
  1,000 messages per business number per month (user messages are free).

Config (all secrets stay local):
    JARVIS_WHATSAPP_ENABLED=1
    JARVIS_WA_PHONE_ID=<test/production phone number id>
    JARVIS_WA_TOKEN=<permanent system-user token>
    JARVIS_WA_TO=<your WhatsApp number, digits + country code>
    JARVIS_WA_VERIFY_TOKEN=<webhook verify token>
    JARVIS_WA_APP_SECRET=<app secret, for X-Hub-Signature-256>
    JARVIS_WA_TEMPLATE_NOTIFY=<optional approved template name>
    JARVIS_WA_TEMPLATE_LANG=en_US
"""
import json
import urllib.error
import urllib.request

from . import config
from .log import log as _mk_log

log = _mk_log("whatsapp")


def configured() -> bool:
    return (config.get_bool("JARVIS_WHATSAPP_ENABLED")
            and bool(config.get("JARVIS_WA_PHONE_ID"))
            and bool(config.get("JARVIS_WA_TOKEN")))


def _post(payload: dict) -> bool:
    phone_id = config.get("JARVIS_WA_PHONE_ID")
    version = config.get("JARVIS_WA_API_VERSION") or "v23.0"
    url = f"https://graph.facebook.com/{version}/{phone_id}/messages"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('JARVIS_WA_TOKEN')}",
        })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        log.warning("WhatsApp send failed (%s): %s", e.code, detail)
        return False
    except Exception as e:
        log.warning("WhatsApp send failed: %s", e)
        return False


def send(text: str, to: str = "") -> bool:
    """Free-form text (only valid inside the 24h customer-service window)."""
    if not configured():
        return False
    target = (to or config.get("JARVIS_WA_TO") or "").strip()
    if not target:
        log.warning("WhatsApp recipient missing (JARVIS_WA_TO)")
        return False
    return _post({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": target,
        "type": "text",
        "text": {"preview_url": False, "body": str(text or "")[:4000]},
    })


def send_template(title: str, message: str, to: str = "") -> bool:
    """Approved utility template with two body parameters (title, message)."""
    name = (config.get("JARVIS_WA_TEMPLATE_NOTIFY") or "").strip()
    if not (configured() and name):
        return False
    target = (to or config.get("JARVIS_WA_TO") or "").strip()
    if not target:
        return False
    return _post({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": target,
        "type": "template",
        "template": {
            "name": name,
            "language": {"code": config.get("JARVIS_WA_TEMPLATE_LANG")
                         or "en_US"},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "text": title[:200]},
                    {"type": "text", "text": message[:900]},
                ],
            }],
        },
    })


def notify(title: str, message: str) -> bool:
    """Proactive notification: template when configured, else text (in-window)."""
    if send_template(title, message):
        return True
    return send(f"*{title}*\n{message}")
