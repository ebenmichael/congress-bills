# -*- coding: utf-8 -*-
"""
Memory efficient code for computing ngrams as a corpus iterator

@author: Eli Ben-Michael
"""

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict, namedtuple
import string
import csv
import os
import re
import argparse
import logging
import six
from six.moves import cPickle as pickle
import zipfile
from gensim.corpora import dictionary
from gensim import matutils
from gensim import models
import numpy as np
from sklearn import feature_extraction
from docLoader import *


# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')


class keydefaultdict(defaultdict):
    """Class to assign the key as the value in a defaultdict"""
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

        
class CorpusIterator(six.Iterator):
    """Class to do tokenization, tagging, stemming, etc. and yield each
        document 1 at a time. Only ever loads one doc into memory"""
    def __init__(self, indir, n, stem=None, stop_words=True, tag=None,
                 tag_pattern=None, punctuation=True,
                 split_clauses=False, outdir=None):
        """Constructor
        Input:
            indir: path to directory of txt files
            n: order of n gram
            stem: {'snowball','porter','lemma',None} stemmer to use
                    Defaults to None.
            stop_words: Boolean. include stopwords. Defaults to True
            tag: {'nltk',None}. POS tagger to use. Defaults 
                    to None
            tag_pattern: list of of tag patterns to allow in simplified form.
                         Defaults to None. if tag_pattern = "default", 
                         use default tag pattern.
            punctuation: Boolean. include punctuation. Defaults to True
            split_clauses: Boolean. Split on clauses
            outdir: directory to write to. Defaults to indir/ngram_results
        """
        self.indir = indir
        # check if directory is zip archive or directory and act accordingly
        if not zipfile.is_zipfile(indir):
            # list the files in the directory
            self.files = sorted([os.path.join(indir, f)
                                 for f in os.listdir(indir)
                                 if os.path.splitext(f)[1] == ".txt"])
            # create directory for results
            if outdir is None:
                outdir = os.path.join(indir,"ngram_results")
            # check if directory exists, if not create direcotry
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            # set zip_corpus to None
            self.zip_corpus = None
        else:
            # files is the namelist of the zip archive
            self.zip_corpus = zipfile.ZipFile(indir)
            self.files = self.zip_corpus.namelist()
            # create directory for results in the directory of the zip archive
            if outdir is None:
                # get the directory of the zip archive
                tmp = os.path.split(indir)[0]
                outdir = os.path.join(tmp, "ngram_results")
            # check if the directory exists, if not create the directory
            if not os.path.exists(outdir):
                os.mkdir(outdir)
        # assign option variables
        self.n = n
        self.stem = stem
        self.stop_words = stop_words
        self.tag = tag
        self.tag_pattern = tag_pattern
        self.punctuation = punctuation
        self.outdir = outdir
        
        # keep an index for the __next__() function
        self.index = 0
 
        # class variable holding default tag patterns and dict for conversion
        # to universal tag set
        self.default_tagpatterns = set(['AN', 'NN', 'VN', 'VV', 'VP', 'NNN',
                                        'AAN', 'ANN', 'NAN', 'NPN', 'VAN',
                                        'VNN', 'VPN', 'ANV', 'NVV', 'VDN',
                                        'VVV', 'VVP'])
        self.default_tagset = set(''.join(self.default_tagpatterns))
        self.tagdict = keydefaultdict(lambda x: x,
                                      {'NN': 'N',
                                       'NNS': 'N',
                                       'NNP': 'N',
                                       'NNPS': 'N',
                                       'JJ': 'A',
                                       'JJR': 'A',
                                       'JJS': 'A',
                                       'VBG': 'A',
                                       'RB': 'A',
                                       'DT': 'D',
                                       'IN': 'P',
                                       'TO': 'P',
                                       'VB': 'V',
                                       'VBD': 'V',
                                       'VBN': 'V',
                                       'VBP': 'V',
                                       'VBZ': 'V',
                                       'MD': 'V',
                                       'RP': 'P'})

        # class variable which contains english stop words as a set
        self.stop = set(stopwords.words('english'))
        
        # set up tagger if tag is not None
        if tag is not None:
            if tag == 'nltk':
                # create a named tuple which holds nltk.pos_tag_sents as
                # tag_sents
                NLTKTagger = namedtuple("NLTKTagger", ["tag_sents", "tag"])
                self.tagger = NLTKTagger(nltk.pos_tag_sents, nltk.pos_tag)
            else:
                # raise a value error if an unsupproted tagger is included
                raise ValueError('Not an available tagger')
        # initialize stemmer if stem is not None
        if stem is not None:
            if stem == 'porter':
                self.stemmer = PorterStemmer()
            elif stem == 'snowball':
                self.stemmer = SnowballStemmer("english")
            elif stem == 'lemma':
                self.stemmer = WordNetLemmatizer()
                # add stem as another name for lemmatize
                self.stemmer.stem = stemmer.lemmatize
            else:
                # raise a value error if a wrong stemmer is chosen
                raise ValueError('Not an available stemmer')
        # set splitting on clauses
        self.split_clauses = split_clauses
        # current clauses
        self.curr_clauses = []
        
    def __len__(self):
        """len function, number of documents"""
        return(len(self.files))
        
    def __next__(self):
        """Next function for iterator"""
        if self.index >= len(self.files):
            raise StopIteration
        # if not splitting on clauses
        if not self.split_clauses:
            # get sentences from file
            sents = doc_sents(self.files[self.index], zipped=self.zip_corpus)
            #get ngrams of doc and yield 
            ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                self.stop_words, self.tag,
                                                self.tag_pattern, self.punctuation)
            self.index += 1
            return(ngrams)
        #if splitting on clauses use the clauses
        else:
            if len(self.curr_clauses) == 0:
                #get the sentences for the current clauses
                self.curr_clauses = doc_sents(self.files[self.index],
                                         zipped = self.zip_corpus,
                                         clauses = True)
                self.index += 1
            #pop one clauses from self.curr_clauses
            sents = self.curr_clauses.pop()
            ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                self.stop_words, self.tag,
                                                self.tag_pattern, self.punctuation)
            return(ngrams)
            
            
    
    def __iter__(self):
        """Iterator, does tokenization,stemming,tagging,etc on a doc before
            returning it"""
        if not self.split_clauses:
            for i, fName in enumerate(sorted(self.files)):
                if i % 100 == 0:
                    logging.info("Computing N-grams for %ith file %s" %(i,fName))            
                #get sentences from file
                sents = doc_sents(fName,zipped = self.zip_corpus)
                #get ngrams of doc and yield 
                ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                self.stop_words, self.tag,
                                                self.tag_pattern, self.punctuation)
                yield(ngrams)
        else:
            for i, fName in enumerate(self.files):
                if i % 100 == 0:
                    logging.info("Computing N-grams for %ith file %s" %(i,fName))  
                #get sentences for clauses
                clauses = doc_sents(fName,zipped = self.zip_corpus,clauses = True)
                for sents in clauses:
                    ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                    self.stop_words, self.tag,
                                                    self.tag_pattern, self.punctuation)
                    yield(ngrams)                    
            

    def custom_ngrams(self,words,n):
        """Faster n gram generation than nltk.ngrams
        Input:
            words: word tokenized sentence
            n: order of ngram
        Output:
            ngrams: list of ngrams
        """
        ngrams = zip(*[words[i:] for i in range(n)])
        return(ngrams)
    
    def word_tokenize(self,words):
        """Faster word tokenization than nltk.word_tokenize
        Input:
            words: a string to be tokenized
        Output:
            tokens: tokenized words
        """
        tokens = re.findall(r"[a-z]+-?[a-z]+", words.lower(),
                            flags = re.UNICODE | re.LOCALE)
        return(tokens)
            
    def ngrams_from_sents(self, sents,n, stem = None, stop_words = True, 
                          tag = None, tag_pattern = None, punctuation = True):
        """Gets the ngrams from a list of sentences
        Input:
            sents: list of sentences as strings
            n: order of n gram
            stem: {'snowball','porter','lemma',None} stemmer to use
                    Defaults to None.
            stop_words: Boolean. include stopwords. Defaults to True
            tag: {'ap','nltk','stanford',None}. POS tagger to use. Defaults 
                    to None
            tag_pattern: list of of tag patterns to allow in simplified form.
                         Defaults to None. if tag_pattern = "default", 
                         use default tag pattern.
            punctuation: Boolean. include punctuation. Defaults to True
        Output:
            ngrams: list of ngrams as "word1-word2" strings
        """
           
        
    
        #tag sentences first
        if tag is not None:
            #tokenize the sentences
            tmp = []
            for sent in sents:
                tmp.append([word.lower() for word in self.word_tokenize(sent)])
            sents = tmp
            if tag == 'nltk':
                # tag words
                tags = self.tagger.tag_sents(sents)
                # extract the tags without the words
                tags = [[self.tagdict[tagWord[1]] for tagWord in tag[i]] for i
                        in range(len(sents))]
            else:
                #raise a value error if an unsupproted tagger is included
                raise ValueError('Not an available tagger')
        #iterate through sentences and get ngrams
        ngrams = []
        for i,words in enumerate(sents):
            if tag is None:
                #if tag is None then word tokenization hasn't happend
                words = self.word_tokenize(words)
            #stem words if stem is not None
            if stem is not None:
                words = [self.stemmer.stem(word) for word in words]
            #join tags and words if tag is not None
            if tag is not None:
                words = ['::'.join(tagWord) for tagWord in zip(words,tags[i])]
            #remove stop words if stop = False
            if not stop_words:
                words = [word for word in words if not word.split("::")[0] in self.stop]
            #remove punctuation if punctuation is false
            if not punctuation:
                pun = string.punctuation
                words = [word for word in words if not word.split("::")[0] in pun]
                
            #get n grams and add to ngrams list
            sent_grams = ["_".join(gram) for gram in 
                            self.custom_ngrams(words,n)]
            #if tag_pattern isn't None, go through sent_grams and only keep those
            #ngrams with the proper tag pattern
            if tag_pattern is not None:
                #assign default tag pattern if tag_pattern == 'default'
                if tag_pattern == 'default':
                    tag_pattern = self.default_tagpatterns
                tmp = []
                #maybe make this a list comprehension?
                for gram in sent_grams:
                    #get tags separately
                    tags_squash = [t.split("::")[1] for t in gram.split("_")]
                    #check if the tag pattern is allowed
                    if ''.join(tags_squash) in tag_pattern:
                        tmp.append(gram)
                sent_grams = tmp
    
                
            ngrams.extend(sent_grams)  
            
        return(ngrams)                
        

class Dictionary(dictionary.Dictionary):
    """Child class of gensim Dictionary, takes in a corpus iterator during
        Initialization"""
        
    def __init__(self,corpus = None):
        """Constructor
        Input:
            corpus: a CorpusIterator
        """
        if corpus is not None:
            #call super constructor
            super(Dictionary,self).__init__(doc for doc in corpus)
        else:
            super(Dictionary,self).__init__()
        
    def update_id2token(self):
        """Creates/updates the id2token dict from the token2id dict"""       
        #get id2token
        self.id2token = dict((v, k) for k, v in six.iteritems(self.token2id))


class GramCorpus(CorpusIterator):
    """Child class of CorpusIterator, gets corpus as bag of words (grams)"""
    
    
    def __init__(self,indir, n, tfidf = False, stem = None, stop_words = True, 
                 tag = None, tag_pattern = None, punctuation = True,
                 threshold = 5, split_clauses = False, outdir = None):
        """Constructor
        Input:
            indir: path to directory of txt files
            n: order of n gram
            tfidf: Boolean. do tfidf
            stem: {'snowball','porter','lemma',None} stemmer to use
                    Defaults to None.
            stop_words: Boolean. include stopwords. Defaults to True
            tag: {'ap','nltk','stanford',None}. POS tagger to use. Defaults 
                    to None
            tag_pattern: list of of tag patterns to allow in simplified form.
                         Defaults to None. if tag_pattern = "default", 
                         use default tag pattern.
            punctuation: Boolean. include punctuation. Defaults to True
            freq: Boolean. Return frequencies. Defaults to False
            threshold: minimum number of documents a gram has to be present in.
                        Defaults to 5.
            split_clauses: Boolean. Split on clauses
            outdir: directory to write to. Defaults to indir/results
        """
        self.dictionary = Dictionary()
        self.threshold = threshold
        #self.dictionary = Dictionary(tmp_corpus)
        #self.dictionary.update_id2token()
        #do thresholding
        #self.dictionary.filter_extremes(no_below=threshold,no_above = 1,
        #                                keep_n = None)
        super(GramCorpus,self).__init__(indir,n,stem,stop_words,tag,
                                        tag_pattern,punctuation,split_clauses,
                                        outdir)
        self.tfidf_bool = tfidf


    def __next__(self):
        """Next function for iterator. same as CorpusIterator but transforms
            to bag of words"""
        if self.index >= len(self.files):
            raise StopIteration
        #if not splitting on clauses
        if not self.split_clauses:
            #get sentences from file
            sents = doc_sents(self.files[self.index],zipped = self.zip_corpus)
            #get ngrams of doc and yield 
            ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                self.stop_words, self.tag,
                                                self.tag_pattern, self.punctuation)
            self.index += 1
            return(self.dictionary.doc2bow(ngrams,allow_update = True))
        #if splitting on clauses use the clauses
        else:
            if len(self.curr_clauses) == 0:
                #get the sentences for the current clauses
                self.curr_clauses = doc_sents(self.files[self.index],
                                         zipped = self.zip_corpus,
                                         clauses = True)
                self.index += 1
            #pop one clauses from self.curr_clauses
            sents = self.curr_clauses.pop()
            ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                self.stop_words, self.tag,
                                                self.tag_pattern, self.punctuation)
            return(self.dictionary.doc2bow(ngrams,allow_update = True))
                        
    
    def __iter__(self):
        """Overidden Iterator, does the same as CorpusIterator but transforms
            to bag of words"""
        if not self.split_clauses:
            for i, fName in enumerate(self.files):
                if i % 100 == 0:
                    logging.info("Computing N-grams for %ith file %s" %(i,fName))            
                #get sentences from file
                sents = doc_sents(fName,zipped = self.zip_corpus)
                #get ngrams of doc and yield 
                ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                self.stop_words, self.tag,
                                                self.tag_pattern, self.punctuation)
                yield(self.dictionary.doc2bow(ngrams,allow_update = True))
        else:
            for i, fName in enumerate(self.files):
                if i % 100 == 0:
                    logging.info("Computing N-grams for %ith file %s" %(i,fName))  
                #get sentences for clauses
                clauses = doc_sents(fName,zipped = self.zip_corpus,clauses = True)
                for sents in clauses:
                    ngrams = self.ngrams_from_sents(sents,self.n,self.stem,
                                                    self.stop_words, self.tag,
                                                    self.tag_pattern, self.punctuation)
                    yield(self.dictionary.doc2bow(ngrams,allow_update = True))                
    def to_sparse(self):
        """Turns the corpus into a sparse matrix"""
        #get sparse matrix
        sparse = matutils.corpus2csc(self,num_docs = len(self.files)).T
        #do thresholding
        sparse = self.threshold_matrix(sparse)
        #do tfidf if True
        if self.tfidf_bool:
            sparse = self.tfidf(sparse)
        #return sparse matrix
        return(sparse)
    
    def threshold_matrix(self,sparse):
        """Does thresholding on a sparse matrix
        Input:
            sparse: sparse matrix to do threhsolding on
        Output:
            sparse: sparse matrix after thresholding
        """
        logging.info("thresholding")
        #convert the sparse matrix to a csc matrix b/c its faster
        sparse = sparse.tocsc()
        #create a boolean mask
        mask = np.zeros(sparse.shape[1],dtype = bool)
        #change the dictionary token2id to account for droped terms
        id2token = {}
        token2id = {}
        count = 0
        #get all columns with greater than threshold doc freq
        for i in range(sparse.shape[1]):
            if sparse[:,i].nnz >= self.threshold:
                mask[i] = True
                token = self.dictionary[i]
                #use count as the new id
                id2token[count] = token
                token2id[token] = count
                #increase count
                count += 1                
        #only keep the columns with greater than threshold doc freq
        mask_flat = np.flatnonzero(mask)        
        sparse = sparse[:,mask_flat]   
        #put these new dicts in the Dictionary
        self.dictionary.id2token = id2token
        self.dictionary.token2id = token2id
        #convert back to csr
        sparse = sparse.tocsr()
        return(sparse)
        
    def tfidf(self,sparse):
        """Returns the tfidf transformation of a sparse matrix
        Input:
            sparse: sparse amtrix to be transformed
        Output:
            sparse: sparse matrix after tfidf transformation
        """
        transform = feature_extraction.text.TfidfTransformer()
        sparse = transform.fit_transform(sparse)
        return(sparse)
