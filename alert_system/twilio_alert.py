from twilio.rest import Client
from .config import *

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