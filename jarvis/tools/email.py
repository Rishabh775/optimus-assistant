"""Email via IMAP read + SMTP send. Works with Gmail/Outlook/etc.
For Gmail, generate an App Password at https://myaccount.google.com/apppasswords
and enable IMAP in Gmail settings.
"""
import os
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime, timedelta


def _creds():
    user = (os.getenv("JARVIS_EMAIL_USER") or "").strip()
    pw = (os.getenv("JARVIS_EMAIL_PASS") or "").strip()
    host = (os.getenv("JARVIS_EMAIL_IMAP") or "").lower()
    # Gmail shows app passwords as "abcd efgh ijkl mnop"; login needs 16 chars.
    if "gmail" in host and len(pw.replace(" ", "")) == 16:
        pw = pw.replace(" ", "")
    return (user, pw)


def _imap():
    user, pw = _creds()
    if not user or not pw:
        return None, None, "email not configured (set JARVIS_EMAIL_USER/PASS)"
    host = os.getenv("JARVIS_EMAIL_IMAP", "imap.gmail.com")
    try:
        m = imaplib.IMAP4_SSL(host)
        m.login(user, pw)
        return m, user, None
    except Exception as e:
        return None, None, f"imap error: {e}"


def _decode(s):
    if s is None:
        return ""
    parts = decode_header(s)
    out = ""
    for text, enc in parts:
        out += text.decode(enc or "utf-8", errors="replace") if isinstance(text, bytes) else text
    return out


def _snippet(msg, n: int = 200) -> str:
    """First n chars of the plain-text body (for at-a-glance triage)."""
    try:
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True)
                if body:
                    txt = body.decode(errors="replace")
                    return " ".join(txt.split())[:n]
    except Exception:
        pass
    return ""


def _fetch_recent(n: int = 10, folder: str = "INBOX", unread_only: bool = False):
    """Return [dict,...] of the newest n messages (or (None, error))."""
    m, user, err = _imap()
    if err:
        return None, err
    try:
        m.select(folder)
        crit = "(UNSEEN)" if unread_only else "ALL"
        typ, data = m.search(None, crit)
        ids = data[0].split()[-n:][::-1]
        items = []
        for mid in ids:
            # PEEK so triaging does NOT mark messages as read
            typ, md = m.fetch(mid, "(FLAGS BODY.PEEK[])")
            flags = md[0][0].decode(errors="replace")
            msg = email.message_from_bytes(md[0][1])
            items.append({
                "id": mid.decode(),
                "unread": "\\Seen" not in flags,
                "from": _decode(msg["From"]),
                "subject": _decode(msg["Subject"]),
                "date": msg["Date"],
                "snippet": _snippet(msg),
            })
        m.logout()
        return items, None
    except Exception as e:
        try:
            m.logout()
        except Exception:
            pass
        return None, f"ERROR: {e}"


def _cfg_list(env: str, default: str = "") -> list:
    raw = os.getenv(env, default)
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


# Sensible defaults; override in .env
_DEFAULT_HIGH = ("invoice", "payment", "urgent", "asap", "deadline",
                 "action required", "verify", "security", "alert", "expir",
                 "interview", "offer", "contract", "overdue")
_DEFAULT_LOW = ("newsletter", "unsubscribe", "promo", "sale", "discount",
                "marketing", "no-reply", "noreply")


def _score_email(item: dict) -> tuple:
    """Deterministic importance score + human-readable reasons."""
    vip = _cfg_list("JARVIS_EMAIL_VIP_SENDERS")
    high = _cfg_list("JARVIS_EMAIL_PRIORITY_KEYWORDS", ",".join(_DEFAULT_HIGH))
    low = _cfg_list("JARVIS_EMAIL_LOW_KEYWORDS", ",".join(_DEFAULT_LOW))
    frm = item["from"].lower()
    text = f"{item['subject']} {item['snippet']}".lower()
    score, reasons = 0, []
    if any(v and v in frm for v in vip):
        score += 100
        reasons.append("VIP sender")
    for kw in high:
        if kw and kw in text:
            score += 20
            reasons.append(kw)
    for kw in low:
        if kw and kw in text:
            score -= 25
            reasons.append(f"-{kw}")
    if item["unread"]:
        score += 10
        reasons.append("unread")
    return score, reasons


def email_triage(n: int = 15, unread_only: bool = False) -> str:
    """Rank recent emails by deterministic importance (VIP senders + keyword
    weights) and explain why. Use for 'what's important in my email'."""
    items, err = _fetch_recent(n=n, unread_only=unread_only)
    if err:
        return err
    if not items:
        return "(no messages)"
    scored = [( _score_email(it)[0], _score_email(it)[1], it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, reasons, it in scored:
        label = "HIGH" if score >= 50 else ("LOW" if score <= -10 else "NORMAL")
        why = f"  why: {', '.join(reasons)}" if reasons else ""
        out.append(f"[{it['id']}] {label} (score {score}) {'UNREAD' if it['unread'] else 'read'} | {it['date']}\n"
                   f"  From: {it['from']}\n  Subj: {it['subject']}\n  {it['snippet']}{why}")
    return "\n\n".join(out)


def email_list(n: int = 10, folder: str = "INBOX", unread_only: bool = False) -> str:
    items, err = _fetch_recent(n=n, folder=folder, unread_only=unread_only)
    if err:
        return err
    out = []
    for it in items:
        out.append(f"[{it['id']}] {'UNREAD' if it['unread'] else 'read  '} | {it['date']} | {it['from']}\n"
                   f"  Subj: {it['subject']}\n  {it['snippet']}")
    return "\n\n".join(out) or "(no messages)"


def email_read(mail_id: str, folder: str = "INBOX") -> str:
    m, user, err = _imap()
    if err:
        return err
    try:
        m.select(folder)
        # PEEK: inspecting an email should not silently mark it read
        typ, data = m.fetch(mail_id.encode(), "(BODY.PEEK[])")
        msg = email.message_from_bytes(data[0][1])
        body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_payload(decode=True).decode(errors="replace")
        return (f"From: {_decode(msg['From'])}\n"
                f"To: {_decode(msg['To'])}\n"
                f"Subject: {_decode(msg['Subject'])}\n"
                f"Date: {msg['Date']}\n\n{body[:6000]}")
    except Exception as e:
        return f"ERROR: {e}"


def email_send(to: str, subject: str, body: str) -> str:
    user, pw = _creds()
    if not user or not pw:
        return "email not configured"
    host = os.getenv("JARVIS_EMAIL_SMTP", "smtp.gmail.com")
    port = int(os.getenv("JARVIS_EMAIL_SMTP_PORT", "587"))
    try:
        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return f"SENT to {to}"
    except Exception as e:
        return f"ERROR: {e}"


def email_unread_count():
    m, user, err = _imap()
    if err:
        return err
    try:
        m.select("INBOX")
        typ, data = m.search(None, "(UNSEEN)")
        n = len(data[0].split())
        m.logout()
        return n
    except Exception as e:
        return f"ERROR: {e}"