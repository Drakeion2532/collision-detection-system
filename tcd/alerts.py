# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Multi-channel emergency alerts: email (SMTP), SMS (Twilio) and a Google Maps link.

Credentials come from environment variables (see .env.example), never from code.
With `alerts.dry_run: true` in config.yaml nothing is sent; the messages are returned
so they can be checked in tests and the demo.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


def compose(inc, cfg: dict, incident_id=None) -> dict:
    loc = cfg["alerts"]["location"]
    link = maps_link(loc["lat"], loc["lon"])
    ts = f"{int(inc.timestamp_s // 60):02d}:{inc.timestamp_s % 60:05.2f}"
    subject = f"[Collision alert] {loc['label']} - confidence {inc.confidence:.0%}"
    body = (f"Possible traffic collision detected.\n"
            f"Incident: {incident_id if incident_id is not None else '-'}\n"
            f"Location: {loc['label']} ({link})\n"
            f"Video time: {ts}  (frame {inc.frame_index})\n"
            f"Model confidence: {inc.confidence:.1%}\n"
            f"Please verify the attached snapshot and dispatch if confirmed.")
    sms = f"Collision alert at {loc['label']} ({inc.confidence:.0%}). Map: {link}"
    return {"subject": subject, "body": body, "sms": sms, "maps": link}


def _snapshot_jpeg(frame):
    if frame is None:
        return None
    import cv2
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def send_email(msg: dict, frame=None) -> str:
    m = EmailMessage()
    m["Subject"], m["From"], m["To"] = msg["subject"], os.environ["ALERT_EMAIL_FROM"], os.environ["ALERT_EMAIL_TO"]
    m.set_content(msg["body"])
    jpg = _snapshot_jpeg(frame)
    if jpg:
        m.add_attachment(jpg, maintype="image", subtype="jpeg", filename="snapshot.jpg")
    with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("SMTP_PORT", 587))) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        s.send_message(m)
    return "email sent"


def send_sms(msg: dict) -> str:
    from twilio.rest import Client
    c = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    c.messages.create(body=msg["sms"], from_=os.environ["TWILIO_FROM"], to=os.environ["ALERT_SMS_TO"])
    return "sms sent"


def send_all(inc, cfg: dict, incident_id=None) -> dict:
    msg = compose(inc, cfg, incident_id)
    if cfg["alerts"].get("dry_run", True):
        return {"dry_run": True, **msg}
    status = {}
    for name, fn in (("email", lambda: send_email(msg, inc.frame)), ("sms", lambda: send_sms(msg))):
        try:
            status[name] = fn()
        except Exception as e:  # one failed channel must not block the others
            status[name] = f"failed: {e}"
    return {"dry_run": False, **msg, "status": status}
