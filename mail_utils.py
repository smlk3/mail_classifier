from imap_tools import MailBox, AND
import datetime

class MailHandler:
    def __init__(self):
        self.mailbox = None

    def connect(self, server, email, password):
        """Mail sunucusuna bağlanır."""
        try:
            self.mailbox = MailBox(server).login(email, password)
            return True, "Bağlantı Başarılı"
        except Exception as e:
            return False, str(e)

    def fetch_latest_emails(self, limit=10, folder='INBOX'):
        """Son mailleri çeker."""
        if not self.mailbox:
            return []
        
        email_list = []
        try:
            self.mailbox.folder.set(folder)
            # Tarihe göre tersten (en yeni en üstte)
            for msg in self.mailbox.fetch(reverse=True, limit=limit):
                email_list.append({
                    "subject": msg.subject,
                    "sender": msg.from_,
                    "date": msg.date_str,
                    "body": msg.text or msg.html, # Text varsa al, yoksa html
                    "id": msg.uid
                })
        except Exception as e:
            print(f"Fetch Error: {e}")
            
        return email_list

    def logout(self):
        if self.mailbox:
            self.mailbox.logout()
