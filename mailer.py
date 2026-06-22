import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config

logger = logging.getLogger("marketpulse.mailer")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def send(subject, html_body):
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = config.GMAIL_ADDRESS
    message["To"] = config.RECIPIENT_EMAIL
    message.attach(MIMEText(html_body, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
        server.sendmail(config.GMAIL_ADDRESS, config.RECIPIENT_EMAIL, message.as_string())
    logger.info("Email sent to %s: %s", config.RECIPIENT_EMAIL, subject)
