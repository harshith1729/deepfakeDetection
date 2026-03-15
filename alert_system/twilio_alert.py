from twilio.rest import Client
import os
from dotenv import load_dotenv

# load .env variables
load_dotenv()

ACCOUNT_SID = os.getenv("ACCOUNT_SID")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
OWNER_PHONE = os.getenv("OWNER_PHONE")
DOOR_NAME = os.getenv("DOOR_NAME")
HOUSE_NAME = os.getenv("HOUSE_NAME")
CITY = os.getenv("CITY")

def send_alert():

    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message = f"""
⚠️ SECURITY ALERT

Deepfake voice attack detected.

Door: {DOOR_NAME}
House: {HOUSE_NAME}
City: {CITY}
"""

    client.messages.create(
        body=message,
        from_=TWILIO_PHONE,
        to=OWNER_PHONE
    )