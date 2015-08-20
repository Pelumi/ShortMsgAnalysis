__author__ = 'Pelumi'

class SMS:

    def __init__(self, id, sender, recipient, text, age, gender, type, time, polarity="NA"):
        self.id = id
        self.sender = sender
        self.recipient = recipient
        self.time = time
        self.text = text
        self.age = age
        self.gender = gender
        self.type = type
        self.time = time
        self.polarity = polarity

    def toDict(self):
        smsDict = {"text": self.text, "id": self.id, "sender": self.sender, "recipient": self.recipient, "time": self.time, "polarity": self.polarity}
        return smsDict

    def toString(self):
        return self.id + "\t" + self.text + "\t" + self.sender  + "\t" + self.recipient  + "\t" +  self.age  + "\t" + self.gender  + "\t" + self.time  + "\t" + self.type + "\t" + self.polarity;