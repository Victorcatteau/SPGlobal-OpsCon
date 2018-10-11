# -*- coding: utf-8 -*-
"""
https://www.reddit.com/r/learnpython/comments/612mik/sending_email_via_python_outlook/
https://msdn.microsoft.com/en-us/library/microsoft.office.interop.outlook.mailitem_properties.aspx
"""


# LIBRARIES
from win32com.client import DispatchWithEvents
import pythoncom
import pandas as pd



#--------------------------------------------------------------------
# MAIN CLASS

class Handler_Class(object):
    def OnNewMailEx(self, receivedItemsIDs):
        # RecrivedItemIDs is a collection of mail IDs separated by a ",".
        # You know, sometimes more than 1 mail is received at the same moment.

        for ID in receivedItemsIDs.split(","):
            mail = outlook.Session.GetItemFromID(ID)
            data = {}

            data["subject"] = mail.Subject
            data["creation_time"] = mail.CreationTime
            data["body"] = mail.Body

            # data["sender"] = mail.Sender
            data["sender_email"] = mail.SenderEmailAddress
            data["sender_name"] = mail.SenderName

            data["to"] = mail.To.split(";")

            print({k:v for k,v in data.items() if k in ["sender_email","subject"]})

            with open("C:/Users/talvesdacosta/Documents/Ekimetrics/11. R&D/3. Listening outlook/mails.txt","a") as file:
                print(data,file = file)




#--------------------------------------------------------------------
# MAIN LOOP


if __name__ == '__main__':
    outlook = DispatchWithEvents("Outlook.Application", Handler_Class)
    pythoncom.PumpMessages()