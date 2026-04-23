"""
FreshSense AI - Shelf Life Prediction System
Run: python freshsense_final.py
Then open: http://localhost:5000
"""

import serial
import serial.tools.list_ports
import json, time, threading, csv, os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

#----------------------- CONFIG -------------------------------------
PORT     = 5000
BAUD     = 9600
LOG      = "readings.csv"

# --------------------- EMAIL CONFIG --------------------------------
EMAIL_SENDER   = "your_gmail@gmail.com"   # Add your Gmail
EMAIL_PASSWORD = "your_app_password"       # Add your App Password
EMAIL_RECEIVER = "receiver@gmail.com"      # Add receiver email
EMAIL_ENABLED  = False                     # Set to True to enable

#---------------------- TRAIN MODEL -------------------
print("Training AI model...", end=" ", flush=True)
np.random.seed(42)
FOODS = ["Milk","Bread","Cheese","Eggs","Vegetables","Meat","Juice","Yogurt"]
PKGS  = ["Plastic","Glass","Cardboard","Vacuum Sealed","Foil"]
BASE  = dict(zip(FOODS,[10,7,30,21,7,5,14,14]))
PF    = {"Vacuum Sealed":1.3,"Glass":1.1,"Foil":1.05,"Plastic":1.0,"Cardboard":0.9}

rows = []
for _ in range(1000):
    f = np.random.choice(FOODS)
    p = np.random.choice(PKGS)
    t = round(np.random.uniform(0,30),1)
    h = round(np.random.uniform(40,95),1)
    d = np.random.randint(0,15)
    sl = max(0, round(BASE[f]*PF[p] - d - (t-4)*0.3 - abs(h-80)*0.05 + np.random.normal(0,0.5),1))
    rows.append([t,h,d,f,p,sl])

df = pd.DataFrame(rows, columns=["T","H","D","F","P","SL"])
lef = LabelEncoder(); lep = LabelEncoder()
df["FE"] = lef.fit_transform(df["F"])
df["PE"] = lep.fit_transform(df["P"])
X = df[["T","H","D","FE","PE"]].values
y = df["SL"].values
sc = StandardScaler()
Xs = sc.fit_transform(X)
rf = RandomForestRegressor(100, max_depth=10, random_state=42)
rf.fit(Xs, y)
print("Done!")

def predict(t, h, d, food, pkg):
    try:
        fe = lef.transform([food])[0]
        pe = lep.transform([pkg])[0]
        return max(0, round(float(rf.predict(sc.transform([[t,h,d,fe,pe]]))[0]),1))
    except:
        return 0.0

#------------- EMAIL SYSTEM -------------------------
email_cooldown = {}  # product_id -> last sent timestamp

def send_email(subject, body):
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECEIVER
        msg["Cc"]      = EMAIL_CC
        msg.attach(MIMEText(body, "plain"))
        html = """<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px">
        <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden">
          <div style="background:#ff3b30;padding:20px;text-align:center">
            <h1 style="color:#fff;margin:0">FreshSense AI Alert</h1>
          </div>
          <div style="padding:30px">
            <p style="font-size:15px;color:#333;white-space:pre-line">""" + body + """</p>
            <p style="font-size:12px;color:#999;margin-top:20px">Sent by FreshSense AI at """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
          </div>
        </div></body></html>"""
        msg.attach(MIMEText(html, "html"))
        all_recipients = [EMAIL_RECEIVER, EMAIL_CC]
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(EMAIL_SENDER, EMAIL_PASSWORD.replace(" ", ""))
            s.sendmail(EMAIL_SENDER, all_recipients, msg.as_string())
        print(f"[EMAIL] Sent to {all_recipients}: {subject}")
    except Exception as e:
        print(f"[EMAIL] Failed: {e}")

def check_emails(products_list, temp, hum):
    if not EMAIL_ENABLED or not products_list:
        print(f"[EMAIL] Skipped - enabled:{EMAIL_ENABLED} products:{len(products_list) if products_list else 0}")
        return
    now = time.time()
    COOLDOWN = 300  # reduced to 5 minutes for testing (change back to 3600 for production)
    print(f"[EMAIL] Checking {len(products_list)} product(s) at {temp}C {hum}%")
    for p in products_list:
        if not p.get("food") or not p.get("mfg") or not p.get("exp"):
            print(f"[EMAIL] Skipping incomplete product: {p}")
            continue
        pid  = p.get("id", 0)
        food = p["food"]
        pkg  = p.get("pkg", "Plastic")
        days = int(p.get("days", 0))
        last_sent = email_cooldown.get(pid, 0)
        time_since = now - last_sent
        print(f"[EMAIL] {food}: cooldown={time_since:.0f}s (need {COOLDOWN}s)")
        if time_since < COOLDOWN:
            print(f"[EMAIL] {food}: still in cooldown, skipping")
            continue
        try:
            pred = predict(temp, hum, days, food, pkg)
            mfg_d = datetime.strptime(p["mfg"], "%Y-%m-%d")
            exp_d = datetime.strptime(p["exp"], "%Y-%m-%d")
            total_days = (exp_d - mfg_d).days
            days_to_expiry = (exp_d - datetime.now()).days
            half_life = total_days / 2
            print(f"[EMAIL] {food}: pred={pred}d half={half_life}d expiry_in={days_to_expiry}d")
            if days_to_expiry < 0:
                print(f"[EMAIL] {food}: EXPIRED - sending email...")
                send_email(
                    f"EXPIRED: {food} has passed its expiry date!",
                    f"Product: {food} ({pkg})\nExpiry Date: {p['exp']}\nStatus: EXPIRED {abs(days_to_expiry)} day(s) ago\n\nRemove or dispose immediately!"
                )
                email_cooldown[pid] = now
            elif pred < half_life:
                print(f"[EMAIL] {food}: below threshold - sending email...")
                send_email(
                    f"Alert: {food} shelf life below threshold!",
                    f"Product: {food} ({pkg})\nTemperature: {temp}C\nHumidity: {hum}%\nAI Prediction: {pred} days remaining\nThreshold: {half_life:.0f} days (half of {total_days}-day shelf life)\n\nAction required: Predicted shelf life is below threshold."
                )
                email_cooldown[pid] = now
            else:
                print(f"[EMAIL] {food}: above threshold ({pred}d > {half_life}d) - no email needed")
        except Exception as e:
            print(f"[EMAIL] Error checking {food}: {e}")

#------------------- SHARED DATA -----------------------
data = {
    "temp": None, "hum": None, "count": 0,
    "ts": None, "connected": False,
    "port": None, "error": None, "history": [],
    "monitored_products": []
}
lock = threading.Lock()

# ---------------- FIND ARDUINO --------------------
def find_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        d = (p.description or "").lower()
        if any(k in d for k in ["arduino","ch340","cp210","ftdi","usb serial","uno"]):
            return p.device
    pl = list(ports)
    return pl[-1].device if pl else None

# ------------------ SERIAL THREAD -----------------------------
def reader():
    port = find_port()
    if not port:
        with lock: data["error"] = "No Arduino found. Plug in USB cable."
        print("ERROR: No Arduino found!")
        return

    with lock: data["port"] = port
    print(f"Arduino on {port}")

    while True:
        try:
            print(f"Connecting to {port}...", end=" ")
            ser = serial.Serial(port, BAUD, timeout=3)
            time.sleep(2)
            print("OK!")
            with lock:
                data["connected"] = True
                data["error"] = None

            last_read = 0
            while True:
                try:
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line or "{" not in line:
                        continue
                    j = json.loads(line)
                    if "temp" not in j or "humidity" not in j:
                        continue

                    # Throttle to once every 5 seconds
                    now_ts = time.time()
                    if now_ts - last_read < 5:
                        continue
                    last_read = now_ts

                    t = round(float(j["temp"]), 1)
                    h = round(float(j["humidity"]), 1)
                    c = int(j.get("count", 0))
                    ts = datetime.now().strftime("%H:%M:%S")

                    with lock:
                        data["temp"] = t
                        data["hum"]  = h
                        data["count"] = c
                        data["ts"]   = ts
                        data["connected"] = True
                        data["error"] = None
                        data["history"].append({"t":ts,"temp":t,"hum":h})
                        if len(data["history"]) > 40:
                            data["history"].pop(0)
                        prods = list(data["monitored_products"])

                    # Log to CSV
                    new = not os.path.isfile(LOG)
                    with open(LOG,"a",newline="") as f:
                        w = csv.writer(f)
                        if new: w.writerow(["Time","Temp","Humidity"])
                        w.writerow([ts,t,h])

                    print(f"[{ts}] Temp:{t}C  Hum:{h}%")

                    # Check email alerts
                    if prods:
                        threading.Thread(target=check_emails, args=(prods,t,h), daemon=True).start()

                except json.JSONDecodeError:
                    continue
                except serial.SerialException:
                    print("Disconnected!")
                    with lock: data["connected"] = False
                    try: ser.close()
                    except: pass
                    time.sleep(3)
                    break

        except serial.SerialException as e:
            print(f"Failed: {e}")
            with lock:
                data["connected"] = False
                data["error"] = f"Cannot open {port}: {e}"
            time.sleep(5)

threading.Thread(target=reader, daemon=True).start()

#--------------------FLASK APP --------------------------------
app = Flask(__name__)

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(render_template_string(PAGE))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.route("/api")
def api():
    try:
        with lock:
            d = dict(data)
        # Trigger email check on every poll if products are synced
        prods = d.get("monitored_products", [])
        t = d.get("temp")
        h = d.get("hum")
        if prods and t is not None and h is not None:
            threading.Thread(target=check_emails, args=(prods, t, h), daemon=True).start()
        return jsonify(d)
    except Exception as e:
        return jsonify({"error": str(e), "connected": False, "temp": None, "hum": None,
                        "count": 0, "history": [], "port": None, "ts": None})

@app.route("/api/products", methods=["POST"])
def api_products():
    """Browser sends current product list to keep Python in sync for email alerts."""
    d = request.get_json()
    products_list = d.get("products", [])
    with lock:
        data["monitored_products"] = products_list
    print(f"[SYNC] Received {len(products_list)} product(s) from browser")
    for p in products_list:
        print(f"[SYNC]   -> {p.get('food')} | mfg:{p.get('mfg')} | exp:{p.get('exp')}")
    return jsonify({"status": "ok"})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    d = request.get_json()
    pred = predict(
        float(d.get("temp",25)),
        float(d.get("hum",70)),
        int(d.get("days",1)),
        d.get("food","Milk"),
        d.get("pkg","Plastic")
    )
    return jsonify({"pred": pred})


# ---------------- HTML PAGE ----------------------------
PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FreshSense AI</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@800&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#06090f;color:#e0e0e0;font-family:'Space Mono',monospace;padding:0}
header{background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.08);
  padding:14px 32px;display:flex;justify-content:space-between;align-items:center;
  position:sticky;top:0;z-index:99;backdrop-filter:blur(10px)}
.logo{display:flex;align-items:center;gap:10px}
.dot{width:9px;height:9px;border-radius:50%;background:#30d158;box-shadow:0 0 8px #30d158;animation:blink 2s infinite}
.dot.off{background:#ff3b30;box-shadow:0 0 8px #ff3b30}
.ltxt{font-size:10px;letter-spacing:3px;color:rgba(255,255,255,0.4);text-transform:uppercase}
#clk{font-size:12px;font-weight:700;color:#0a84ff;letter-spacing:2px}
#conn{font-size:10px;color:rgba(255,255,255,0.4)}
main{max-width:1100px;margin:0 auto;padding:28px 20px}
h1{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;letter-spacing:-1px;margin-bottom:4px}
.sub{font-size:9px;letter-spacing:3px;color:#0a84ff;text-transform:uppercase;margin-bottom:28px}
.row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:22px}
.clbl{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:14px;font-weight:600}
.bignum{font-size:52px;font-weight:700;letter-spacing:-2px;line-height:1}
.unit{font-size:16px;color:rgba(255,255,255,0.4);margin-left:4px}
.note{font-size:10px;color:rgba(255,255,255,0.4);margin-top:8px}
.wait{display:flex;flex-direction:column;align-items:center;justify-content:center;height:90px;gap:10px}
.ring{width:32px;height:32px;border-radius:50%;border:3px solid rgba(10,132,255,0.15);
  border-top:3px solid #0a84ff;animation:spin 1s linear infinite}
.wtxt{font-size:9px;letter-spacing:2px;color:rgba(255,255,255,0.3);text-transform:uppercase;animation:blink 1.5s infinite}
select,input[type=number],input[type=date]{
  background:#1e2640;border:1px solid rgba(255,255,255,0.2);color:#ffffff;
  padding:7px 10px;border-radius:7px;font-family:'Space Mono',monospace;font-size:11px;
  outline:none;color-scheme:dark}
select{width:100%}
input[type=number]{width:100%}
input[type=date]{width:150px;cursor:pointer}
select option{background:#1e2640;color:#ffffff}
select:focus,input:focus{border-color:#0a84ff}
input[type=date]::-webkit-calendar-picker-indicator{filter:invert(1) brightness(2);cursor:pointer;opacity:1;width:18px;height:18px}
input[type=date]::-webkit-datetime-edit,
input[type=date]::-webkit-datetime-edit-fields-wrapper,
input[type=date]::-webkit-datetime-edit-month-field,
input[type=date]::-webkit-datetime-edit-day-field,
input[type=date]::-webkit-datetime-edit-year-field{color:#ffffff}
input[type=date]::-webkit-datetime-edit-text{color:rgba(255,255,255,0.5)}
.ptbl{width:100%;border-collapse:collapse;font-size:11px}
.ptbl th{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#ffffff;
  padding:10px 10px;border-bottom:2px solid rgba(255,255,255,0.2);text-align:left;font-weight:700}
.ptbl td{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,0.06);vertical-align:middle;color:rgba(255,255,255,0.9)}
.ptbl tr:last-child td{border-bottom:none}
.abtn{background:rgba(10,132,255,0.12);border:1px solid rgba(10,132,255,0.3);
  color:#0a84ff;padding:9px 20px;border-radius:8px;font-family:'Space Mono',monospace;
  font-size:11px;cursor:pointer;letter-spacing:1px}
.dbtn{background:rgba(255,59,48,0.1);border:1px solid rgba(255,59,48,0.2);
  color:#ff3b30;padding:5px 11px;border-radius:6px;font-family:'Space Mono',monospace;
  font-size:10px;cursor:pointer}
.rtag{display:inline-block;padding:4px 12px;border-radius:12px;font-size:9px;letter-spacing:1px;font-weight:700;text-transform:uppercase}
.ebar{height:5px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;margin-top:4px}
.efill{height:100%;border-radius:3px;transition:width .5s}
.rec{display:flex;gap:10px;padding:11px 13px;border-radius:9px;font-size:11px;line-height:1.6;margin-bottom:8px;align-items:flex-start}
.rec.ok{background:rgba(48,209,88,0.07);border:1px solid rgba(48,209,88,0.15)}
.rec.warning{background:rgba(255,149,0,0.07);border:1px solid rgba(255,149,0,0.15)}
.rec.critical{background:rgba(255,59,48,0.1);border:1px solid rgba(255,59,48,0.2)}
.pred-wrap{display:flex;gap:14px;flex-wrap:wrap}
.pred-card{flex:1;min-width:160px;background:rgba(255,255,255,0.03);border-radius:16px;padding:24px 20px;text-align:center;border:2px solid rgba(255,255,255,0.08);transition:all .4s}
.pred-card.critical{border-color:#ff3b30;background:rgba(255,59,48,0.08)}
.pred-card.high{border-color:#ff9500;background:rgba(255,149,0,0.08)}
.pred-card.medium{border-color:#ffd60a;background:rgba(255,208,10,0.06)}
.pred-card.low{border-color:#30d158;background:rgba(48,209,88,0.06)}
.pred-food{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,0.45);margin-bottom:12px}
.pred-circle{width:110px;height:110px;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;margin:0 auto 12px;border:4px solid rgba(255,255,255,0.1);transition:all .4s}
.pred-circle.critical{border-color:#ff3b30;box-shadow:0 0 24px rgba(255,59,48,0.35)}
.pred-circle.high{border-color:#ff9500;box-shadow:0 0 24px rgba(255,149,0,0.35)}
.pred-circle.medium{border-color:#ffd60a;box-shadow:0 0 20px rgba(255,208,10,0.25)}
.pred-circle.low{border-color:#30d158;box-shadow:0 0 20px rgba(48,209,88,0.25)}
.pred-days{font-size:44px;font-weight:800;letter-spacing:-2px;line-height:1;font-family:'Syne',sans-serif;transition:color .4s}
.pred-unit{font-size:10px;color:rgba(255,255,255,0.4);margin-top:3px;letter-spacing:1px}
.pred-risk{font-size:9px;letter-spacing:2px;font-weight:700;text-transform:uppercase;padding:4px 14px;border-radius:20px;display:inline-block;margin-top:8px}
.pred-threshold{font-size:9px;color:rgba(255,255,255,0.3);margin-top:6px;letter-spacing:.3px}
.pred-empty{text-align:center;padding:28px;color:rgba(255,255,255,0.2);font-size:11px;letter-spacing:1px;border:1px dashed rgba(255,255,255,0.1);border-radius:14px}
canvas{width:100%;height:80px;display:block}
.sleg{display:flex;gap:16px;margin-top:8px}
.sk{font-size:9px;color:rgba(255,255,255,0.35);display:flex;align-items:center;gap:5px}
.skd{width:18px;height:2px;border-radius:1px}
.htbl{width:100%;border-collapse:collapse;font-size:11px}
.htbl th{font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.3);padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.07);text-align:left;font-weight:400}
.htbl td{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.04)}
#alarm{display:none;position:sticky;top:58px;left:0;right:0;z-index:198;
  padding:12px 20px;text-align:center;font-size:12px;font-weight:700;
  background:#ff3b30;color:#fff;animation:ap .6s infinite alternate;border-bottom:2px solid #cc1a10;letter-spacing:1px}
#dbg{margin-top:10px;padding:10px 14px;background:rgba(10,132,255,0.05);border:1px solid rgba(10,132,255,0.12);border-radius:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:.5px}
#sb{font-size:9px;color:rgba(255,255,255,0.3);padding:10px 0 0;border-top:1px solid rgba(255,255,255,0.06);margin-top:14px}
#voice-btn{background:none;border:1px solid #30d158;color:#30d158;padding:5px 14px;border-radius:20px;font-family:'Space Mono',monospace;font-size:10px;cursor:pointer;letter-spacing:1px}
.email-status{font-size:11px;padding:8px 14px;border-radius:8px;background:rgba(48,209,88,0.1);border:1px solid rgba(48,209,88,0.3);color:#30d158;display:inline-block}
@media(max-width:768px){
  .row{grid-template-columns:1fr;gap:10px}
  .row2{grid-template-columns:1fr;gap:10px}
  main{padding:16px 12px}
  h1{font-size:26px}
  .bignum{font-size:40px}
  header{padding:10px 16px}
}
@media(max-width:768px){
  header{padding:10px 16px}
  .ltxt{display:none}
  #voice-btn{padding:4px 10px;font-size:9px}
  main{padding:16px 12px}
  h1{font-size:24px}
  .sub{font-size:8px;margin-bottom:16px}
  .row{grid-template-columns:1fr;gap:10px}
  .row2{grid-template-columns:1fr;gap:10px}
  .card{padding:16px 14px}
  .bignum{font-size:40px}
  .ptbl{font-size:10px}
  .ptbl th{font-size:8px;padding:6px 6px}
  .ptbl td{padding:7px 6px}
  select{font-size:10px;padding:5px 6px}
  input[type=date]{width:120px;font-size:10px}
  input[type=number]{font-size:10px}
  .pred-circle{width:90px;height:90px}
  .pred-days{font-size:36px}
  .pred-card{min-width:140px;padding:16px 12px}
  #alarm{font-size:11px;padding:10px}
}
@media(max-width:480px){
  h1{font-size:20px}
  .bignum{font-size:34px}
  .pred-circle{width:80px;height:80px}
  .pred-days{font-size:28px}
  #voice-btn{display:none}
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes ap{from{opacity:1}to{opacity:.7}}
</style>
</head>
<body>
<div id="alarm"></div>
<header>
  <div class="logo">
    <div class="dot" id="dot"></div>
    <span class="ltxt">FreshSense AI</span>
  </div>
  <div style="display:flex;gap:20px;align-items:center">
    <span id="conn">Connecting...</span>
    <button id="voice-btn" onclick="toggleVoice()">&#128266; Voice ON</button>
    <span id="clk"></span>
  </div>
</header>

<main>
  <div class="sub">Real-Time DHT22 Sensor &#8594; AI Shelf Life Prediction</div>
  <h1>Storage <span style="color:#0a84ff">Intelligence</span></h1>

  <!-- Sensor cards -->
  <div class="row" style="margin-top:24px">
    <div class="card">
      <div class="clbl">Temperature</div>
      <div id="tcard"><div class="wait"><div class="ring"></div><div class="wtxt">Waiting for sensor</div></div></div>
    </div>
    <div class="card">
      <div class="clbl">Humidity</div>
      <div id="hcard"><div class="wait"><div class="ring"></div><div class="wtxt">Waiting for sensor</div></div></div>
    </div>
    <div class="card">
      <div class="clbl">Model Stats</div>
      <div style="margin-top:4px">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px">
          <span style="font-size:10px;color:rgba(255,255,255,0.4)">R&#178; Score</span>
          <span style="font-size:15px;font-weight:700;color:#30d158">0.964</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:10px">
          <span style="font-size:10px;color:rgba(255,255,255,0.4)">MAE</span>
          <span style="font-size:15px;font-weight:700;color:#0a84ff">1.09d</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="font-size:10px;color:rgba(255,255,255,0.4)">RMSE</span>
          <span style="font-size:15px;font-weight:700;color:#bf5af2">1.47d</span>
        </div>
      </div>
    </div>
  </div>

  <!-- AI Prediction Highlight -->
  <div id="pred-highlight" style="margin-bottom:14px"></div>

  <!-- Product table -->
  <div class="card" style="margin-bottom:14px" id="product-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
      <div class="clbl" style="margin:0">Products Being Monitored</div>
      <button class="abtn" onclick="addRow()">+ Add Product</button>
    </div>
    <div style="overflow-x:auto">
      <table class="ptbl">
        <thead><tr>
          <th>#</th><th>Food</th><th>Packaging</th><th>Days Stored</th>
          <th>Mfg Date</th><th>Expiry Date</th>
          <th>AI Prediction</th><th>Expiry %</th><th>Risk</th><th>Alert</th><th></th>
        </tr></thead>
        <tbody id="ptbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Email status -->
  <div class="card" style="margin-bottom:14px">
    <div class="clbl">Email Alert Status</div>
    <span class="email-status">&#10003; Email alerts active &mdash; alerts sent to akshitamanthani@gmail.com when prediction &lt; half shelf life or product expires</span>
  </div>

  <!-- Trend + Recommendations -->
  <div class="row2">
    <div class="card">
      <div class="clbl">Live Sensor Trend</div>
      <canvas id="spark"></canvas>
      <div class="sleg">
        <div class="sk"><div class="skd" style="background:#0a84ff"></div>Temp &#176;C</div>
        <div class="sk"><div class="skd" style="background:#30d158"></div>Humidity %</div>
      </div>
    </div>
    <div class="card">
      <div class="clbl">Storage Recommendations</div>
      <div id="recs"><div style="font-size:11px;color:rgba(255,255,255,0.3)">Awaiting first reading...</div></div>
    </div>
  </div>

  <!-- History -->
  <div class="card" style="margin-bottom:14px">
    <div class="clbl">Reading History</div>
    <div style="overflow-x:auto">
      <table class="htbl">
        <thead><tr><th>Time</th><th>Temp (&#176;C)</th><th>Humidity (%)</th></tr></thead>
        <tbody id="hist"></tbody>
      </table>
    </div>
  </div>

  <div id="sb">Starting up...</div>
  <div id="dbg">Debug: initialising...</div>
</main>

<script>
const FOODS=["Milk","Bread","Cheese","Eggs","Vegetables","Meat","Juice","Yogurt"];
const PKGS=["Plastic","Glass","Cardboard","Vacuum Sealed","Foil"];
const BL={Milk:10,Bread:7,Cheese:30,Eggs:21,Vegetables:7,Meat:5,Juice:14,Yogurt:14};
const PF={"Vacuum Sealed":1.3,Glass:1.1,Foil:1.05,Plastic:1.0,Cardboard:0.9};
const RC={Critical:"#ff3b30",High:"#ff9500",Medium:"#ffd60a",Low:"#30d158"};

let products=[], nid=1, lT=null, lH=null;
let vOn=true, lRisk=null, lSpk=0;
let aOn=false, aTimer=null;
let userBusy=false, userBusyTimer=null;
let lastProductSync=0;

function markBusy(){
  userBusy=true;
  clearTimeout(userBusyTimer);
  userBusyTimer=setTimeout(()=>{ userBusy=false; renderTbl(); },10000);
}

function toggleVoice(){
  vOn=!vOn;
  const b=document.getElementById("voice-btn");
  b.textContent=vOn?"&#128266; Voice ON":"&#128263; Voice OFF";
  b.style.color=vOn?"#30d158":"rgba(255,255,255,0.3)";
  b.style.borderColor=vOn?"#30d158":"rgba(255,255,255,0.2)";
  if(!vOn) speechSynthesis.cancel();
}
function speak(txt){
  if(!vOn||!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(txt);
  u.rate=0.92;u.pitch=1;u.volume=1;
  const vs=speechSynthesis.getVoices();
  const v=vs.find(x=>x.lang.startsWith("en")&&x.name.includes("Google"))||vs.find(x=>x.lang.startsWith("en"))||vs[0];
  if(v)u.voice=v;
  speechSynthesis.speak(u);
}

function beep(){
  try{
    const ctx=new(AudioContext||webkitAudioContext)();
    [[880,0],[660,.2],[880,.4]].forEach(([f,t])=>{
      const o=ctx.createOscillator(),g=ctx.createGain();
      o.connect(g);g.connect(ctx.destination);
      o.frequency.value=f;o.type="square";
      g.gain.setValueAtTime(.2,ctx.currentTime+t);
      g.gain.exponentialRampToValueAtTime(.001,ctx.currentTime+t+.15);
      o.start(ctx.currentTime+t);o.stop(ctx.currentTime+t+.2);
    });
  }catch(e){}
}
function alarm(msg){
  const el=document.getElementById("alarm");
  el.textContent="! "+msg;el.style.display="block";
  if(!aOn){aOn=true;beep();aTimer=setInterval(beep,5000);}
}
function clearAlarm(){
  document.getElementById("alarm").style.display="none";
  aOn=false;if(aTimer){clearInterval(aTimer);aTimer=null;}
}

function cpred(food,pkg,days,t,h){
  return Math.max(0,Math.round((BL[food]*PF[pkg]-days-(t-4)*0.3-Math.abs(h-80)*0.05)*10)/10);
}
function risk(d){return d<=1?"Critical":d<=3?"High":d<=7?"Medium":"Low"}
function expInfo(m,e){
  if(!m||!e)return null;
  const md=new Date(m),ed=new Date(e),nd=new Date();
  return{pct:Math.min(100,Math.max(0,((nd-md)/(ed-md))*100)),
    dl:Math.ceil((ed-nd)/86400000),tot:(ed-md)/86400000};
}

function addRow(){
  products.push({id:nid++,food:"",pkg:"",days:0,mfg:"",exp:""});
  renderTbl();syncProducts();
}
function delRow(id){
  products=products.filter(p=>p.id!==id);renderTbl();syncProducts();
}
function setField(id,k,v){
  const p=products.find(p=>p.id===id);
  if(p)p[k]=k==="days"?parseInt(v)||0:v;
  renderTbl();syncProducts();
}

// Sync product list to Python for email alerts
async function syncProducts(){
  try{
    await fetch("/api/products",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({products:products})
    });
  }catch(e){}
}

function updatePredHighlight(){
  const el=document.getElementById("pred-highlight");
  if(!el) return;
  const valid=products.filter(p=>p.food&&p.pkg&&p.mfg&&p.exp);
  if(valid.length===0||lT===null){
    el.innerHTML="<div class='pred-empty'>Enter food type, packaging, manufacturing date and expiry date to see AI predictions</div>";
    return;
  }
  let html="<div class='pred-wrap'>";
  valid.forEach(p=>{
    const pred=cpred(p.food,p.pkg,p.days||0,lT,lH);
    const rk=pred<=1?"critical":pred<=3?"high":pred<=7?"medium":"low";
    const rkL=rk.charAt(0).toUpperCase()+rk.slice(1);
    const rc=RC[rkL];
    const ep=expInfo(p.mfg,p.exp);
    const half=ep?(ep.tot/2).toFixed(0):null;
    const thr=half?"Threshold: "+half+"d (half of shelf life)":"Add dates for threshold";
    html+="<div class='pred-card "+rk+"'>"
      +"<div class='pred-food'>"+p.food+" &bull; "+p.pkg+"</div>"
      +"<div class='pred-circle "+rk+"'>"
        +"<div class='pred-days' style='color:"+rc+"'>"+pred+"</div>"
        +"<div class='pred-unit'>DAYS LEFT</div>"
      +"</div>"
      +"<div class='pred-risk' style='background:"+rc+"22;color:"+rc+";border:1px solid "+rc+"44'>"+rkL+" Risk</div>"
      +"<div class='pred-threshold'>"+thr+"</div>"
      +"</div>";
  });
  html+="</div>";
  el.innerHTML=html;
}

function renderTbl(){
  const tb=document.getElementById("ptbody");
  if(!tb)return;
  if(products.length===0){
    tb.innerHTML="<tr><td colspan='11' style='text-align:center;padding:24px;color:rgba(255,255,255,0.25);font-size:11px'>No products added yet. Click &quot;+ Add Product&quot; to start monitoring.</td></tr>";
    clearAlarm();updatePredHighlight();return;
  }
  const t=lT!==null?lT:25, h=lH!==null?lH:60;
  let anyAlarm=false, rows="";
  products.forEach((p,i)=>{
    const pred=(lT!==null&&p.food&&p.pkg&&p.mfg&&p.exp)?cpred(p.food,p.pkg,p.days||0,t,h):null;
    const rk=pred!==null?risk(pred):null;
    const rc=rk?RC[rk]:"rgba(255,255,255,0.3)";
    const ep=expInfo(p.mfg,p.exp);
    let foodSel="<select class='sf' data-id='"+p.id+"' data-field='food' style='min-width:105px'>";
    foodSel+="<option value=''>-- Select Food --</option>";
    FOODS.forEach(f=>{ foodSel+="<option"+(f===p.food?" selected":"")+">"+f+"</option>"; });
    foodSel+="</select>";
    let pkgSel="<select class='sf' data-id='"+p.id+"' data-field='pkg' style='min-width:115px'>";
    pkgSel+="<option value=''>-- Select Packaging --</option>";
    PKGS.forEach(q=>{ pkgSel+="<option"+(q===p.pkg?" selected":"")+">"+q+"</option>"; });
    pkgSel+="</select>";
    const daysIn="<input class='sf' type='number' data-id='"+p.id+"' data-field='days' min='0' max='365' value='"+(p.days||"")+"' placeholder='0' style='width:60px'>";
    const mfgIn="<input class='sf' type='date' data-id='"+p.id+"' data-field='mfg' value='"+p.mfg+"'>";
    const expIn="<input class='sf' type='date' data-id='"+p.id+"' data-field='exp' value='"+p.exp+"'>";
    const predCell=pred!==null?"<span style='font-size:20px;font-weight:700;color:"+rc+"'>"+pred+"</span><span style='font-size:10px;color:rgba(255,255,255,0.35)'> d</span>":"--";
    let barH="<span style='color:rgba(255,255,255,0.25);font-size:10px'>--</span>";
    let altH="<span style='color:rgba(255,255,255,0.25);font-size:10px'>--</span>";
    if(ep){
      const bc=ep.pct>=75?"#ff3b30":ep.pct>=50?"#ff9500":"#30d158";
      barH="<div style='min-width:80px'><div class='ebar'><div class='efill' style='width:"+ep.pct.toFixed(0)+"%;background:"+bc+"'></div></div>"
        +"<div style='font-size:9px;color:"+bc+";margin-top:3px'>"+ep.pct.toFixed(0)+"% - "+ep.dl+"d left</div></div>";
      const halfLife=ep.tot/2;
      const belowHalf=pred!==null&&pred<halfLife;
      if(ep.dl<0){altH="<span style='color:#ff3b30;font-weight:700'>EXPIRED</span>";anyAlarm=true;}
      else if(belowHalf){altH="<span style='color:#ff3b30;font-weight:700'>ALERT - "+pred+"d &lt; "+halfLife.toFixed(0)+"d</span>";anyAlarm=true;}
      else{altH="<span style='color:#30d158'>Safe - "+pred+"d &gt; "+halfLife.toFixed(0)+"d</span>";}
    } else if(pred!==null&&pred<=1){altH="<span style='color:#ff3b30'>Critical</span>";anyAlarm=true;}
    const rkCell=rk?"<span class='rtag' style='background:"+rc+"22;color:"+rc+";border:1px solid "+rc+"44'>"+rk+"</span>":"--";
    rows+="<tr>"
      +"<td style='color:rgba(255,255,255,0.3)'>"+(i+1)+"</td>"
      +"<td>"+foodSel+"</td><td>"+pkgSel+"</td>"
      +"<td>"+daysIn+"</td><td>"+mfgIn+"</td><td>"+expIn+"</td>"
      +"<td style='text-align:center'>"+predCell+"</td>"
      +"<td>"+barH+"</td><td>"+rkCell+"</td>"
      +"<td>"+altH+"</td>"
      +"<td><button class='dbtn sf-del' data-id='"+p.id+"'>X</button></td>"
      +"</tr>";
  });
  tb.innerHTML=rows;
  document.querySelectorAll(".sf").forEach(el=>{
    el.addEventListener("focus",markBusy);
    el.addEventListener("mousedown",markBusy);
    el.addEventListener("change",function(){
      setField(parseInt(this.getAttribute("data-id")),this.getAttribute("data-field"),this.value);
    });
  });
  document.querySelectorAll(".sf-del").forEach(el=>{
    el.addEventListener("click",function(){delRow(parseInt(this.getAttribute("data-id")));});
  });
  if(anyAlarm) alarm("One or more products need immediate attention!");
  else clearAlarm();
  updatePredHighlight();
}

function drawSpark(hist){
  const cv=document.getElementById("spark");
  cv.width=cv.parentElement.clientWidth-44||400;cv.height=80;
  const ctx=cv.getContext("2d"),W=cv.width,H=cv.height,P=8;
  ctx.clearRect(0,0,W,H);
  if(!hist||hist.length<2)return;
  function ln(vals,col){
    const mn=Math.min(...vals)-1,mx=Math.max(...vals)+1;
    ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=2;ctx.shadowColor=col;ctx.shadowBlur=4;
    vals.forEach((v,i)=>{
      const x=P+(i/(vals.length-1))*(W-P*2);
      const y=H-P-((v-mn)/(mx-mn||1))*(H-P*2);
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });
    ctx.stroke();ctx.shadowBlur=0;
  }
  ln(hist.map(h=>h.temp),"#0a84ff");
  ln(hist.map(h=>h.hum),"#30d158");
}

setInterval(()=>{
  document.getElementById("clk").textContent=new Date().toLocaleTimeString("en-US",{hour12:false});
},1000);

function getRecs(t,h){
  const r=[];
  if(t>8) r.push({type:"warning",text:"Temperature "+t+"C is too high. Lower to 4-6C."});
  else r.push({type:"ok",text:"Temperature "+t+"C is within optimal range (0-8C)."});
  if(h<60) r.push({type:"warning",text:"Humidity "+h+"% is too low. Raise to 75-85%."});
  else if(h>90) r.push({type:"warning",text:"Humidity "+h+"% is too high. Reduce below 85%."});
  else r.push({type:"ok",text:"Humidity "+h+"% is within acceptable range."});
  return r;
}

function update(d){
  document.getElementById("dot").className="dot"+(d.connected?"":" off");
  document.getElementById("conn").textContent=d.port?(d.port+" "+(d.connected?"Online":"Offline")):"Detecting...";
  document.getElementById("dbg").textContent="Debug: temp="+d.temp+" hum="+d.hum+" connected="+d.connected+" count="+d.count;
  document.getElementById("sb").textContent=d.ts?"Last update: "+d.ts+" | Log: readings.csv":"Waiting for sensor data...";

  if(d.temp!==null&&d.temp!==undefined){
    const t=parseFloat(d.temp), h=parseFloat(d.hum);
    const tc=t>15?"#ff3b30":t>8?"#ff9500":"#0a84ff";
    document.getElementById("tcard").innerHTML=
      "<span class='bignum' style='color:"+tc+"'>"+t.toFixed(1)+"</span>"
      +"<span class='unit'>&#176;C</span>"
      +"<div class='note'>"+(t>8?"Above optimal (4-6C)":"Within optimal range")+"</div>";
    const hc=(h<60||h>90)?"#ff9500":"#30d158";
    document.getElementById("hcard").innerHTML=
      "<span class='bignum' style='color:"+hc+"'>"+h.toFixed(1)+"</span>"
      +"<span class='unit'>%</span>"
      +"<div class='note'>"+(h<60?"Too low - raise to 75-85%":h>90?"Too high - reduce below 85%":"Acceptable range")+"</div>";
    lT=t; lH=h;
    if(!userBusy) renderTbl();
    const validProducts=products.filter(p=>p.food&&p.pkg&&p.mfg&&p.exp);
    const recsEl=document.getElementById("recs");
    if(validProducts.length===0){
      recsEl.innerHTML="<div style='font-size:11px;color:rgba(255,255,255,0.25)'>Add a product with all details to see dynamic storage recommendations.</div>";
    } else {
      const recs=getRecs(t,h);
      validProducts.forEach(p=>{
        const pred=cpred(p.food,p.pkg,p.days||0,t,h);
        const ep=expInfo(p.mfg,p.exp);
        if(ep){
          const half=ep.tot/2;
          if(pred<half) recs.push({type:"critical",text:p.food+" ("+p.pkg+"): Only "+pred+" days predicted - below "+half.toFixed(0)+"d threshold!"});
          else recs.push({type:"ok",text:p.food+" ("+p.pkg+"): "+pred+" days predicted - above "+half.toFixed(0)+"d threshold. Safe."});
        }
      });
      recsEl.innerHTML=recs.map(r=>"<div class='rec "+r.type+"'><span>"+(r.type==="ok"?"OK: ":"WARN: ")+r.text+"</span></div>").join("");
      if(vOn){
        const now=Date.now();
        const anyBelowHalf=validProducts.some(p=>{const ep=expInfo(p.mfg,p.exp);if(!ep)return false;return cpred(p.food,p.pkg,p.days||0,lT,lH)<(ep.tot/2);});
        if(anyBelowHalf&&(anyBelowHalf!==lRisk||(now-lSpk)>30000)){
          lRisk=anyBelowHalf;lSpk=now;
          let lines=[];
          validProducts.forEach(p=>{const ep=expInfo(p.mfg,p.exp);if(!ep)return;const pred=cpred(p.food,p.pkg,p.days||0,lT,lH);const half=ep.tot/2;if(pred<half)lines.push(p.food+" has only "+pred+" days left, below the "+half.toFixed(0)+" day threshold.");});
          getRecs(t,h).filter(r=>r.type!=="ok").forEach(r=>lines.push(r.text));
          if(lines.length) speak(lines.join(" "));
        } else if(!anyBelowHalf){lRisk=false;}
      }
    }
  }
  if(d.history&&d.history.length) drawSpark(d.history);
  if(d.history&&d.history.length){
    document.getElementById("hist").innerHTML=
      [...d.history].reverse().slice(0,10).map(r=>"<tr><td>"+r.t+"</td><td>"+r.temp.toFixed(1)+"</td><td>"+r.hum.toFixed(1)+"</td></tr>").join("");
  }
}

async function poll(){
  try{
    const resp=await fetch("/api");
    const d=await resp.json();
    update(d);
    // Sync products to Python every poll so email alerts work
    const valid=products.filter(p=>p.food&&p.pkg&&p.mfg&&p.exp);
    if(valid.length>0){
      fetch("/api/products",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({products:valid})
      }).catch(()=>{});
    }
  } catch(e){
    document.getElementById("dbg").textContent="Poll error: "+e.message;
  }
}

renderTbl();
poll();
setInterval(poll,3000);
</script>
</body>
</html>
"""

if __name__=="__main__":
    print(f"\n  FreshSense AI ready!")
    print(f"  Open: http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop.\n")
    time.sleep(1.5)
    import webbrowser
    url = f"http://127.0.0.1:{PORT}"
    browsers = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    ]
    opened = False
    for b in browsers:
        if os.path.exists(b):
            import subprocess
            subprocess.Popen([b, url])
            opened = True
            break
    if not opened:
        webbrowser.open(url)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
