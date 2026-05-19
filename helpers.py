import os
import mimetypes
import smtplib
from email.message import EmailMessage


def send_email(subject, body, from_email, to_email, attachment=None):
    """
    Send an email with optional attachment.
    
    Args:
        subject (str): Email subject (mandatory)
        body (str): Email body content (mandatory)
        from_email (str): Sender email address (mandatory)
        to_email (str): Recipient email address (mandatory)
        attachment (str, optional): Path to file to attach. If None, no attachment is sent.
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    email_password = os.getenv("EMAIL_PASSWORD")
    
    if not email_password:
        print("EMAIL FAILED: EMAIL_PASSWORD environment variable not set.")
        return False
    
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(body)
        
        # Add attachment if provided
        if attachment:
            if not os.path.exists(attachment):
                print(f"EMAIL FAILED: Attachment file not found: {attachment}")
                return False
            
            with open(attachment, "rb") as f:
                file_data = f.read()
            
            file_name = os.path.basename(attachment)
            # Detect MIME type based on file extension
            mime_type, _ = mimetypes.guess_type(attachment)
            
            if mime_type:
                maintype, subtype = mime_type.split("/", 1)
            else:
                # Default to octet-stream for unknown types
                maintype, subtype = "application", "octet-stream"
            
            msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=file_name)
        
        # Port 587 + starttls() is the most robust method for bypassing standard SMTP blocks
        print("Connecting to SMTP server...")
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(from_email, email_password)
            smtp.send_message(msg)
            print("Email sent successfully!")
        
        return True
    
    except smtplib.SMTPAuthenticationError:
        print("EMAIL FAILED: Authentication Error. Ensure you are using a Gmail App Password, not your normal password.")
        return False
    except Exception as e:
        print(f"EMAIL FAILED: {e}")
        return False

