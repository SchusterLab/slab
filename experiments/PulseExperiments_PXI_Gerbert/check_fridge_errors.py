# Import needed libraries
import json
import requests
import time
import sys
import smtplib, ssl
import numpy as np
import os

# this code assumes you are working with the new BF temperature controller

# how often the BF4 bot will spam messages before shutting off (so it doesn't spam infinite times)
MAX_ERRORS = 4

#hardcode channels on BF temperature controller that you want to track, and max temperature for each channel
#on BF4 for example, 50K plate is channel 1, 4K plate is channel 2, 1K plate is channel 5, and base plate is channel 8
channels = {
    1: {"label": "55K",
        "ABORT_temp": 90},
    2: {"label": "4K",
        "ABORT_temp": 5},
    5: {"label": "STILL",
        "ABORT_temp": 1.2},
    8: {"label": "MXC",
        "ABORT_temp": 0.20},
}

# This used to email people too, but then gmail changed its privacy settings so that no longer works
def send_email(message_string, receiver_email="schusterbf4error@gmail.com"):
    """This functoin lets you send an email containing message_string to receiver_email
    If you can get google to let you log in to your email through a script without throwing a hissy fit"""
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "schusterbf4error@gmail.com"  # Enter your address
    receiver_email = receiver_email  # Enter receiver address
    password = "dilfridgeist00hot!" # enter password if email you have set up for this
    message = """\
                Subject: BF4 ERROR!
                
                ERROR: """ + message_string

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def post_message_to_slack(text, channel, blocks=None):
    """Posts text to slack channel (DMs also count as a slack channel). Code based off slack API."""

    # the bot token that ids your scipt to slack as a trustworthy.
    # Slack doesn't let you hardcode it on code posted to github, which is why you have to set it as an environment variable
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


def alert_error(channel_nr, temperature, receiver_email_list=["schusterbf4error@gmail.com"], slack_ch_list=[]):
    """function that sends slack messages/emails people if a channel goes above a certain temperature"""

    try:
        error_string = "{} is at {} K".format(channel_nr, np.round(temperature, 3))
    except:
        error_string = "{} temperature overrange, or can't read temperature".format(channel_nr)
    print(error_string)

    # commented out because google threw a hissy fit at me
    # for email in receiver_email_list:
    #     send_email(error_string, receiver_email=email)
    #     email security makes this fail

    for s_ch in slack_ch_list:
        post_message_to_slack(error_string, s_ch, blocks=None)


def alert_com_breakdown(receiver_email_list=["schusterbf4error@gmail.com"], slack_ch_list=[]):
    """Posts message if can't read temperature any more"""
    error_string = "can't get temperature data"
    print(error_string)
    # try:
    #     for email in receiver_email_list:
    #         send_email(error_string, receiver_email=email)
    # except:
    #     print("email not working")
    for s_ch in slack_ch_list:
        post_message_to_slack(error_string, s_ch, blocks=None)




receiver_email_list= ["schusterbf4error@gmail.com", "andrei.vrajitoarea@gmail.com", "bsaxberg@uchicago.edu", "glcroberts@uchicago.edu"]
slack_channels = ['UB2UYK6FP', 'U0179LMURHA','U1Q9UU3HT', 'USW7K9V9P', '#bf4','U041WQZFT']
# the slack "channel" for DMS is called "member ID". In slack, click on the profile of the person you want to send the DM to, then click on the "..." menu, and there should be an option to "copy member ID"

if __name__ == "__main__":
    nb_errors = 0
    while nb_errors < len(channels)*MAX_ERRORS:

        # try to get temperature data using functions from BF temperature controller API
        try:
            # get most recent temperature reading (temperature controller cycles through reading the different channels)
            req = requests.get('http://192.168.14.212:5001/channel/measurement/latest', timeout=30)
        except:
            # if this doesn't work, alert that there has been a communication breakdown
            alert_com_breakdown(["glcroberts@uchicago.edu"], [])

        # check which channel we just read data for
        data = req.json()
        nb = data["channel_nr"]

        # if temperature for that channel is higher than our flag, post errors!
        if data["temperature"] == None or data["temperature"] > channels[nb]["ABORT_temp"]:
            alert_error(channels[nb]["label"], data["temperature"], receiver_email_list, slack_channels)
            nb_errors += 1
        time.sleep(10)
