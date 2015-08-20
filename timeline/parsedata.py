__author__ = 'Pelumi'

import xml.etree.ElementTree as ET
from sms import SMS
import pprint
from solrsearch import searchManager
from datetime import datetime
import pytz
from sentanal.preprocessing.utils import ManageLexicon
from sentanal.preprocessing.const import const


NUSSMSDataset = "/Users/Pelumi/Google Drive/BrisMscProject/DataSets/smsCorpus_en_2012.04.30_all.xml"

tree = ET.parse(NUSSMSDataset)
root = tree.getroot()
local_tz = pytz.timezone('Asia/Tokyo')

allSMS = {}
allSMSDict = {}
count = 0

saveData = ManageLexicon()

def parseTime(timeString):
    if 'AM' in timeString or 'PM' in timeString:
        date = datetime.strptime(timeString, '%m/%d/%Y %I:%M:%S %p')
    elif len(timeString) == 19 or len(timeString) == 17:
        date = datetime.strptime(timeString, '%Y.%m.%d %H:%M:%S')
    elif len(timeString) == 16 or len(timeString) == 15 or len(timeString) == 14:
        date = datetime.strptime(timeString, '%Y.%m.%d %H:%M')
        # elif len(timeString)>20:
        #   date = datetime.strptime(timeString, '%m/%d/%Y %I:%M:%S %p')
    else:
        date = 'NA'

    return local_tz.localize(date)


def parseXML(count):
    solr = searchManager()

    for message in root.findall('message'):
        id = message.get('id')
        text = message.find('text').text
        source = message.find('source')
        sender = source.find('srcNumber').text
        uprofile = source.find('userProfile')
        age = uprofile.find('age').text
        gender = uprofile.find('gender').text
        destination = message.find('destination')
        recipient = destination.find('destNumber').text
        mProfile = message.find('messageProfile')
        time = mProfile.get('time').strip()
        type = mProfile.get('type')

        newSms = SMS(id=id, text=text, sender=sender, recipient=recipient, age=age, gender=gender, time=time,
                     type=type, )

        # print text, sender, recipient,  age, gender, time, type

        if sender in allSMS:
            if time == 'unknown':
                continue
            count += 1

            date = parseTime(time)
            if date == 'NA':
                print 'found faffdate: ', time
                continue
            currentUserSmsList = allSMS[sender]
            currentUserSmsList.append(newSms)
            allSMS[sender] = currentUserSmsList


            currentSmsDictList = allSMSDict[sender]
            currentSmsDictList.append(newSms.toDict())
            allSMSDict[sender] = currentSmsDictList
            #solr.addDocuments(id=id, sender=sender, recipient=recipient, text=text, date=date, polarity='NA')

            if recipient in allSMS:
                print 'recipeint has been a sender', recipient

        else:
            if time == 'unknown':
                continue
            count += 1

            date = parseTime(time)
            if date == 'NA':
                print 'found faffdate: ', time
                continue
            userSmsList = []
            userSmsList.append(newSms)
            allSMS[sender] = userSmsList

            currentSmsDictList = []
            currentSmsDictList.append(newSms.toDict())
            allSMSDict[sender] = currentSmsDictList
            #solr.addDocuments(id=id, sender=sender, recipient=recipient, text=text, date=date, polarity='NA')

    print 'processing complete, a total of: ', len(allSMS), ' unique people.', ' Total messages is: ', count
   # for sender, mesages in allSMS.items():
    #    print sender, ' sent a total of : ', len((mesages)), ' messages'

    for sender, mesages in allSMSDict.items():
        print sender, ' sent a total of : ', len((mesages)), ' messages'
        print(mesages)
        #saveData.savedata(mesages, const.classified_sms_dir+sender)


parseXML(count)
