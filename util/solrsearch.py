__author__ = 'Pelumi'


from urllib2 import *
import solr


class searchManager:
    connection = solr.SolrConnection('http://localhost:8983/solr')
    # connection = urlopen(
    #                 'http://localhost:8983/solr/select?q=*&wt=python')
    # response = eval(connection.read())
    #
    #
    # print response['response']['numFound'], "documents found."
    #
    # # Print the name of each document.
    #
    # for document in response['response']['docs']:
    #   print "  Name =", document['name']


    def addDocuments(self):

        # add a document to the index
        doc = dict(
            id=789,
            title='Lucene in Action',
            author=['Erik Hatcher', 'Otis Gospodneti'],
            )
        doc = {"id": 34567, "title": "Mbuoe"}
        searchManager.connection.add(doc)
#        searchManager.connection.add(doc, commit=True)


    def searchDocuments(self, keyword):
        # do a search
        response = searchManager.connection.query('title:',keyword)
        for hit in response.results:
            print hit['title']


search = searchManager()
search.addDocuments()