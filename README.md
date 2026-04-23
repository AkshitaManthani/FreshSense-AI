# FreshSense AI
### AI-Based Dynamic Shelf Life Prediction System for Packaged Foods

## Overview
FreshSense is a real-time shelf life prediction system that uses a DHT22 
sensor connected to Arduino Uno to monitor temperature and humidity. 
A Random Forest AI model predicts remaining shelf life of packaged foods 
based on actual storage conditions or simulated data (Demo Mode).

## Features
- Live temperature and humidity monitoring
- AI shelf life prediction (R2 = 0.964)
- Multiple product monitoring
- Visual, audio and voice alerts
- Email alerts via Gmail
- Mobile responsive dashboard
- CSV data logging

## Demo Mode
If Arduino is not connected, FreshSense automatically switches to Demo Mode.
In this mode, simulated temperature and humidity data are generated so the system can still run and demonstrate AI predictions.

## Hardware Required
- Arduino Uno
- DHT22 temperature and humidity sensor
- USB cable
- Jumper wires

## Wiring
- DHT22 + (red) → Arduino 5V
- DHT22 out (yellow) → Arduino Digital Pin 2
- DHT22 - (black) → Arduino GND

## Software Setup
1. Install Python 3.11 from python.org
2. Install required packages:
   pip install pyserial flask numpy pandas scikit-learn
3. Upload dht_sensor.ino to Arduino using Arduino IDE
4. py -3.11 freshsense_final.py
5. Open browser: http://localhost:5000

## Email Alerts Setup
1. Go to myaccount.google.com/apppasswords
2. Create an App Password
3. Update EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER in freshsense_final.py
4. Set EMAIL_ENABLED = True

## Troubleshooting

### ModuleNotFoundError (numpy, pandas, etc.)
Run:
pip install numpy pandas flask scikit-learn pyserial

### Python Version Issues
Ensure Python 3.11 is installed:
py -3.11 --version

### File Not Found Error
Navigate to project folder before running:
cd path/to/project

### Arduino Not Detected
- Check USB connection
- Ensure correct COM port
- If not found, system will run in Demo Mode

### Serial Port Error
Close Arduino IDE Serial Monitor before running the Python script

## Project Structure
- freshsense_final.py — Main Python application
- dht_sensor.ino — Arduino sketch for DHT22 sensor
- start_monitor.bat — Windows launcher

## Model Performance
- Algorithm: Random Forest Regression
- R2 Score: 0.964
- MAE: 1.09 days
- RMSE: 1.47 days
- Training samples: 1000

## College
Vignan's Institute of Management and Technology for Women
B.Tech 3rd Year — CSE (AI & ML)