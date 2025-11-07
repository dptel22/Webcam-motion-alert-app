import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import imghdr

load_dotenv()

password = os.getenv("GOOGLE_APP_PASSWORD")
sender = 'devapp2005@gmail.com'
receiver = "dptel07@gmail.com"

def send_email(image_path):
    print("Emailing the client")
    email_message = EmailMessage()
    email_message["Subject"] = "Intruder Detected"
    email_message["From"] = sender
    email_message["To"] = receiver
    email_message.set_content(
        "An intruder has been detected. View the attached image and take necessary action."
    )

    with open(image_path, "rb") as file:
        content = file.read()
        image_type = imghdr.what(None, content) or 'jpeg'

    email_message.add_attachment(content, maintype="image", subtype=image_type, filename=os.path.basename(image_path))

    with smtplib.SMTP("smtp.gmail.com", 587) as gmail:
        gmail.starttls()
        gmail.login(sender, password)
        gmail.send_message(email_message)
        gmail.quit()
    print("Email sent to the client")

if __name__ == "__main__":
    send_email(image_path="images/30.png")
