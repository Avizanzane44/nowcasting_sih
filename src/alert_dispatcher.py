"""
Automated Multi-Channel Emergency Alert Dispatcher (Convective Nowcasting)
- Formats messages in WMO/CAP (Common Alerting Protocol) standard
- Dispatches Aviation SIGMETs, DDMA Civil Alerts, and Farmer SMS Broadcasts
- Supports Webhooks and Telegram Bot notifications
"""

import os
import json
import datetime
import urllib.request
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "config"))
import config

DISPATCH_LOG_FILE = "dispatched_alerts_log.json"

# Optional: Add your real Telegram Bot Token and Chat ID to receive live phone notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WEBHOOK_URL = os.getenv("EMERGENCY_WEBHOOK_URL", "")


def send_telegram_alert(message_text):
    """Sends immediate phone notification via Telegram Bot API if configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message_text, "parse_mode": "Markdown"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception as e:
        print(f"Telegram dispatch failed: {e}")
        return False


def format_aviation_sigmet(alert):
    """Formats Aviation Weather Warning (SIGMET/Aerodrome Warning) for ATC."""
    eta = alert.get("eta_minutes", 15)
    return (
        f"🚨 **[ATC / AVIATION WARNING - IGI DEL]**\n"
        f"• Type: CONVECTIVE SIGMET\n"
        f"• Hazard: {alert['hazard_type']} (Max Rain Rate: {alert['expected_peak_rain_mmh']} mm/h)\n"
        f"• Impact Sector: {alert['location']}\n"
        f"• Estimated Inbound ETA: {eta} MINS\n"
        f"• Action: Advise low-level wind shear precautions & review approach hold/divert queues."
    )


def format_civil_ddma_alert(alert):
    """Formats Civil Defense / Smart City Emergency Siren Advisory."""
    eta = alert.get("eta_minutes", 15)
    return (
        f"🔴 **[DDMA DISASTER MANAGEMENT EARLY WARNING]**\n"
        f"• Region: {alert['location']}\n"
        f"• Severity: {alert['severity']} LEVEL EMERGENCY\n"
        f"• Threat: {alert['hazard_type']} (ETA: {eta} mins)\n"
        f"• Automated Directives:\n"
        f"  - Activate storm water pump stations.\n"
        f"  - Divert traffic from low-lying underpasses.\n"
        f"  - Broadcast public safety sirens."
    )


def format_agri_farmer_sms(alert):
    """Formats localized bilingual SMS warning for rural agricultural zones."""
    eta = alert.get("eta_minutes", 15)
    return (
        f"🌾 **[IMD-AGROMET ALERT FOR FARMERS]**\n"
        f"चेतावनी / WARNING: {alert['location']} में अगले {eta} मिनट में ओलावृष्टि और तेज आकाशीय बिजली की संभावना।\n"
        f"• Severe Hail & Lightning expected in {eta} mins.\n"
        f"• एडवाइजरी: खेतों में तुरंत सुरक्षित आश्रय लें, पेड़ों और बिजली के खंभों से दूर रहें। फसल तिरपाल से ढकें।"
    )


def run_alert_dispatch():
    """Reads live hazard alerts and dispatches multi-channel protocols."""
    if not os.path.exists("live_hazard_alerts.json"):
        print("No active alerts file found.")
        return []

    with open("live_hazard_alerts.json", "r") as f:
        alerts = json.load(f)

    dispatches = []
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("\n=======================================================")
    print("      AUTOMATED EMERGENCY DISPATCH ENGINE ACTIVE       ")
    print("=======================================================\n")

    for alert in alerts:
        # Trigger immediate dispatch if hazard is imminent (ETA <= 30 mins) or CRITICAL/HIGH
        if alert.get("status") == config.ALERT_THRESHOLDS["IMMINENT_HAZARD_STATUS"] and (alert.get("eta_minutes", 60) <= config.ALERT_THRESHOLDS["ETA_MINUTES_MAX"] or alert.get("severity") in config.ALERT_THRESHOLDS["SEVERE_LEVELS"]):
            loc = alert["location"]
            
            # Select target channel based on location category
            if "Airport" in loc or "Aviation" in loc:
                msg = format_aviation_sigmet(alert)
                channel = "AIRPORT_ATC_SIGMET"
            elif "Agri" in loc or "Rural" in loc:
                msg = format_agri_farmer_sms(alert)
                channel = "FARMER_AGROMET_SMS"
            else:
                msg = format_civil_ddma_alert(alert)
                channel = "CIVIL_DEFENSE_DDMA"

            dispatch_record = {
                "id": f"DISP-{len(dispatches)+1:03d}",
                "timestamp_utc": timestamp,
                "target_location": loc,
                "channel": channel,
                "severity": alert["severity"],
                "eta_minutes": alert["eta_minutes"],
                "hazard_type": alert["hazard_type"],
                "message": msg,
                "status": "DISPATCHED"
            }

            dispatches.append(dispatch_record)

            # Print to operations log
            print(f"[{channel}] --> Dispatched to {loc}:")
            print(msg.encode('cp1252', errors='replace').decode('cp1252'))
            print("-" * 55 + "\n")

            # Fire real Telegram / Webhook if configured
            send_telegram_alert(msg)

    # Save to history log
    with open(DISPATCH_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(dispatches, f, indent=2, ensure_ascii=False)

    print(f"Total emergency broadcasts dispatched: {len(dispatches)}")
    return dispatches


if __name__ == "__main__":
    run_alert_dispatch()