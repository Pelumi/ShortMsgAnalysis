import solr
from sms import SMS
import re

class searchManager:
    connection = solr.SolrConnection('http://localhost:8983/solr', debug=True)
# create a connection to a solr server
  #  s = solr.SolrConnection('http://localhost:8983/solr', debug=True)

    # # add a document to the index
    # connection.add(id=1, title=['Erik Hatcher', 'Otis Gospodneti', 'Lucene tutorial', 'lucene created solr'], sms_ss='the sample lucene sms',
    #       polarity_ss='negative', recepient_ss='respid')
    # connection.commit()
    #
    # # do a search
    # response = connection.query('title:lucene')
    # for hit in response.results:
    #     print hit['title']


    def addDocuments(self, id, sender, text, recipient, polarity, date="NA"):
        # add a document to the index
        searchManager.connection.add(id=id, title=text, polarity_ss=polarity, recipient_ss=recipient, sender_ss=sender, date_dt=date)
        #searchManager.connection.commit()

    def searchDocuments(self, keyword, sender='*', polarity='*' ):
        # do a search
        reqString = 'title:'+keyword + ' AND ' + 'sender_ss:' + sender + ' AND ' + 'polarity_ss:' + polarity #+ ' sort=id desc'
        response = searchManager.connection.query(reqString) # 'sender_ss:88c87ecdd9a98fb95e05fe1cfc47a278'
        for hit in response.results:
            #print hit
            print hit['id'], '-->',hit['sender_ss'][0], '-->>',  hit['title'][0], '-->>', hit['recipient_ss'][0], '-->>', hit['polarity_ss'][0], '-->>', search.getPyDay(hit['date_dt'])

    def clearSolrIndex(self):
        searchManager.connection.delete(queries='*')
        #make url call below
        #http://localhost:8983/solr/collection1/update?stream.body=%3Cdelete%3E%3Cquery%3E*:*%3C/query%3E%3C/delete%3E

    def getPyDay(self, dateStr):
        print(dateStr)
        #print(dateStr[:10])
        print dateStr.strftime("%Y-%m-%d")

search = searchManager()
#searchManager.connection.commit()
#search.addDocuments(sender='fde8b6f3df6d88f6ff13979b18b093e6', text='I dont understand :) your message.', recipient='fde8b6f3dfe6', polarity='positive')

search.searchDocuments(keyword='k', sender='fde8b6f3df6d88f6ff13979b18b093e6')


