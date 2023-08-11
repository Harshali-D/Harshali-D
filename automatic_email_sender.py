

import time
from email.message import EmailMessage
import ssl
import smtplib

me='harshali982002@gmail.com'
password= #contact me to see it work real time 
friend= #enter any email adress

subject='this is automated'
body='hello from pluto '

em = EmailMessage()
em['From']=me
em['To']=friend
em['Subject']=subject
em.set_content(body)

context=ssl.create_default_context()

while True:
    with smtplib.SMTP_SSL('smtp.gmail.com',465,context=context) as smtp:
        smtp.login(me,password)
        smtp.sendmail(me,friend,em.as_string())
    time.sleep(0.1)
    print("email sent !")