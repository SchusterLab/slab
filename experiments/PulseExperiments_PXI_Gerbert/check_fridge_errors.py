# Import needed libraries
import json
import requests
import time
import sys
import smtplib, ssl
import numpy as np
import os


def send_email(message_string, receiver_email="schusterbf4error@gmail.com"):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "schusterbf4error@gmail.com"  # Enter your address
    receiver_email = receiver_email  # Enter receiver address
    password = "dilfridgeist00hot!"  # input("Type your password and press enter: ")
    message = """\
Subject: BF4 ERROR!

ERROR: """ + message_string

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def post_message_to_slack(text, channel, blocks=None):
    slack_token = os.environ["SLACK_BOT_TOKEN"]
    slack_channel = channel
    slack_icon_emoji = ':cool:'
    slack_user_name = 'BF4 bot'

    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': slack_token,
        'channel': slack_channel,
        'text': text,
        'icon_emoji': slack_icon_emoji,
        'username': slack_user_name,
        'blocks': json.dumps(blocks) if blocks else None
    }).json()


def error_condition(channel_nr, temperature, receiver_email_list=["schusterbf4error@gmail.com"], slack_ch_list=[]):
    error_string = "{} is at {} K".format(channel_nr, np.round(temperature, 3))
    print(error_string)
    for email in receiver_email_list:
        send_email(error_string, receiver_email=email)
    for s_ch in slack_ch_list:
        post_message_to_slack(error_string, s_ch, blocks=None)


def error_get(receiver_email_list=["schusterbf4error@gmail.com"], slack_ch_list=[]):
    error_string = "can't get temperature data"
    print(error_string)
    for email in receiver_email_list:
        send_email(error_string, receiver_email=email)
    for s_ch in slack_ch_list:
        post_message_to_slack(error_string, s_ch, blocks=None)


NB_ERRORS = 0

channels = {
    1: {"label": "55K",
        "ABORT_temp": 42},
    2: {"label": "4K",
        "ABORT_temp": 3.5},
    5: {"label": "STILL",
        "ABORT_temp": 1},
    8: {"label": "MXC",
        "ABORT_temp": 0.20},
}

receiver_email_list= ["schusterbf4error@gmail.com", "andrei.vrajitoarea@gmail.com", "bsaxberg@uchicago.edu", "glcroberts@uchicago.edu"]
#receiver_email_list = ["schusterbf4error@gmail.com"]
#slack_channels = ['UB2UYK6FP']
slack_channels = ['UB2UYK6FP', 'U0179LMURHA','U1Q9UU3HT', 'USW7K9V9P', '#bf4','U041WQZFT']

if __name__ == "__main__":

    while NB_ERRORS < len(channels)*2:
        try:
            req = requests.get('http://192.168.14.212:5001/channel/measurement/latest', timeout=10)
        except:
            error_get(receiver_email_list, slack_channels)
        data = req.json()
        nb = data["channel_nr"]
        if data["temperature"] == None or data["temperature"] > channels[nb]["ABORT_temp"]:
            error_condition(channels[nb]["label"], data["temperature"], receiver_email_list, slack_channels)
            NB_ERRORS += 1
        time.sleep(10)