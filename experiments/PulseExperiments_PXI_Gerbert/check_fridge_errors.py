# Import needed libraries
import json
import requests
import time
import sys
import smtplib, ssl
import numpy as np

def send_email(message_string, receiver_email="schusterbf4error@gmail.com" ):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "schusterbf4error@gmail.com"  # Enter your address
    receiver_email = receiver_email  # Enter receiver address
    password = "dilfridgeist00hot!" #input("Type your password and press enter: ")
    message = """\
Subject: BF4 ERROR!

ERROR: """ + message_string

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def error_condition(channel_nr, temperature, receiver_email_list=["schusterbf4error@gmail.com"]):
    error_string = "{} is at {} K".format(channel_nr, np.round(temperature,3))
    print(error_string)
    for email in receiver_email_list:
        send_email(error_string, receiver_email=email)

def error_get(receiver_email_list=["schusterbf4error@gmail.com"]):
    error_string = "can't get temperature data"
    print(error_string)
    for email in receiver_email_list:
        send_email(error_string, receiver_email=email)


NB_ERRORS = 0

channels = {
    1: {"label": "55K",
        "ABORT_temp": 55},

    2: {"label": "4K",
        "ABORT_temp": 3.5},
    5: {"label": "STILL",
        "ABORT_temp": 1},
    8: {"label": "MXC",
        "ABORT_temp": 0.011},
}

# receiver_email_list= ["schusterbf4error@gmail.com", "andrei.vrajitoarea@gmail.com", "bsaxberg@uchicago.edu", "glcroberts@uchicago.edu"]
receiver_email_list = ["schusterbf4error@gmail.com"]


if __name__ == "__main__":
    while NB_ERRORS < len(channels):
        try:
            req = requests.get('http://192.168.14.212:5001/channel/measurement/latest', timeout=10)
        except:
            error_get(receiver_email_list)
        data = req.json()
        nb = data["channel_nr"]
        if data["temperature"] == None or data["temperature"] > channels[nb]["ABORT_temp"]:
            error_condition(channels[nb]["label"], data["temperature"], receiver_email_list)
            NB_ERRORS += 1
        time.sleep(10)