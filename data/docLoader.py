# -*- coding: utf-8 -*-
"""
Loads documents and returns the sentences in the documents
@author: Eli
"""

from nltk import sent_tokenize
import os
import six
import codecs
import zipfile


def doc_sents_string(doc):
    """Returns the sentences in a document
    Input:
        doc: the document in string form
    Output:
        sents: a list of the sentences in the document
    """
    # clean any html tags
    # doc = BeautifulSoup(doc).get_text()
    # use sent_tokenize and return
    sents = sent_tokenize(doc)
    return(sents)
    

def doc_sents(docFile,encode = "utf-8-sig",zipped = None,clauses = False):
    """Returns the sentences in a text file
    Input:
        docFile: filepath for a text file
        encode: the encoding of the text file, defaults to utf8
        zipped: a ZipFile object to read fPath from if the directory is 
                zipped. Defaults to None
        clauses: Boolean. Whether to split by clauses
    Output:
        sents: a list of sentences in the text file
    """
    #check version of python and act accordingly with the unicode
    if six.PY2:
        if zipped is None:
            # read the text file and return the sentences
            with codecs.open(docFile,'r',encode) as f:
                doc = f.read()
                sents = doc_sents_string(doc)
                return(sents)
        #if zipped isn't None, use zipped to open the files
        else:
            with zipped.open(docFile) as f:
                doc = codecs.decode(f.read(),encode)
                sents = doc_sents_string(doc)
                return(sents)
    elif six.PY3:
        if zipped is None:
            #read the text file and return the sentences
            with open(docFile,'r',encoding = encode) as f:
                doc = f.read()
                sents = doc_sents_string(doc)
                return(sents)                
        else:
            #if zipped isn't None, use zipped to open the files
            with zipped.open(docFile) as f:
                doc = codecs.decode(f.read(),encode)
                sents = doc_sents_string(doc)
                return(sents)

def corpus_sents(corpus,encoding="utf8"):
    """Returns the sentences in each text file in a corpus
    Input:
        corpus: a list of filepaths to the text files
        encoding: the encoding of the text files, defaults to utf8
    Output:
        sents: a dictionary of lists of sentences in each text file.
                sents[fileName] = list of sentences
    """
    sents = {}
    #iterate through the documents in the corpus
    for doc in corpus:
        #get the file name w/o extension
        path, fileName = os.path.split(doc)
        fName = os.path.splitext(fileName)[0]
        sents[fName] = doc_sents(doc,encode = encoding)
    return(sents)
    
def dir_sents(directory,encoding = "utf8"):
    """Returns the sentences in each text file in a directory
    Input:
        directory: filepath to the directory
        encoding: the encoding of the text files, defaults to utf8
    Output:
        sents: a dictionary of lists of sentences in each text file.
                sents[fileName] = list of sentences
    """
    #get a list of the files in a directory
    corpus = os.listdir(directory)
    #join the file path to the directory to the file names
    for i in range(len(corpus)):
        corpus[i] = os.path.join(directory,corpus[i])
    #get the sentences and return
    sents = corpus_sents(corpus)
    return(sents)
    
