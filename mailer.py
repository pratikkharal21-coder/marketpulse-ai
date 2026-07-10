import logging
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config

logger = logging.getLogger("marketpulse.mailer")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def send(subject, html_body, inline_images=None):
    message = MIMEMultipart("related")
    message["Subject"] = subject
    message["From"] = config.GMAIL_ADDRESS
    message["To"] = config.RECIPIENT_EMAIL

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html"))
    message.attach(alt)

    for cid, image_bytes in (inline_images or {}).items():
        image = MIMEImage(image_bytes)
        # MIMEImage already sniffs the real subtype (png/jpeg/...) from the bytes for
        # Content-Type; match the filename extension to it too instead of hardcoding .png --
        # the real_world_image visual is a JPEG, so it was previously mislabeled.
        image.add_header("Content-ID", f"<{cid}>")
        image.add_header("Content-Disposition", "inline", filename=f"{cid}.{image.get_content_subtype()}")
        message.attach(image)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
        server.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
        server.sendmail(config.GMAIL_ADDRESS, config.RECIPIENT_EMAIL, message.as_string())
    logger.info("Email sent to %s: %s (%d inline image(s))", config.RECIPIENT_EMAIL, subject, len(inline_images or {}))
