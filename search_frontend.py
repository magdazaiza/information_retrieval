import os
os.system("pip install -q pyspark")
os.system("pip install -U -q PyDrive")
os.system("apt-get update -qq")
os.system("apt install openjdk-8-jdk-headless -qq")
os.system("pip install -q graphframes")
os.system("pip install fsspec")
os.system("pip install gcsfs")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'
# spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'
# !wget -N -P $spark_jars $graphframes_jar
# os.system("wget -N -P $spark_jars $graphframes_jar")

from flask import Flask, request, jsonify
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from graphframes import *
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
from inverted_index_gcp import *
import os
import math
import numpy as np
import builtins
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
TUPLE_SIZE=6
TF_MASK=2**16-1


def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list

def calculate_DL(index):
  DL = defaultdict()
  for t,loc in index.posting_locs.items():
    if loc == []:
      continue
    pl = read_posting_list(index,t)
    for id,tf in pl :
        DL[id]=DL.get(id,0)+tf
  return DL

def cosine_similarity(query,index, DL):
  results= {}
  for term in query:
    pl = read_posting_list(index,term)
    for doc_id,tf in pl:
      results[doc_id] = results.get(doc_id,0)+tf
  for doc_id,similarity in results.items():
    results[doc_id] = similarity * (1/len(query)) * (1/DL[doc_id])
  return results

def get_top(sim_dict,N=100):
    return sorted([(doc_id,builtins.round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

def merge_results(title_scores,body_scores,title_weight=0.5,text_weight=0.5,N = 100):    
    dictionary = {}
    body_keys = body_scores.keys()
    title_keys = title_scores.keys()
    docs = set (list(title_keys) + list(body_keys))

    for doc in docs :
        if doc in  body_keys and doc in title_keys :
          score = body_scores[doc]*text_weight + title_scores[doc]*title_weight
        elif doc in  body_keys and doc not in title_keys :
          score = body_scores[doc]*text_weight 
        elif doc not in body_keys and doc in title_keys :
          score = title_scores[doc]*title_weight
        dictionary[doc] = score
    lst = sorted(dictionary.items(), key=lambda item: item[1] , reverse=True)[:N] 
    return lst

def get_relevant(query,index):
  dic={}
  for term in query:
    for doc_id,tf in read_posting_list(index,term):
      dic[doc_id]=dic.get(doc_id,0)+tf
  return dic

def get_relevant_from_body(query,index):
    cs= cosine_similarity(query,index,DL)
    return cs

def get_relevant_from_title(query,index):
    dic={}
    for term in query:
      for doc_id,tf in read_posting_list(index,term):
        dic[doc_id]=dic.get(doc_id,0)+1
    return dic

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)

try:
    if os.environ["208220756"] is not None:
        pass  
except:
   os.system('gsutil -m cp gs://208220756/postings_gcp/* "/content"') #title
   os.system('gsutil -m cp gs://208220756/postings_gcp_body/* "/content"') #body
   os.system('gsutil -m cp gs://208220756/postings_gcp_anchor/* "/content"') #anchor
   os.system('gsutil -m cp gs://208220756/Doc_Len "/content"') #DL
   os.system('gsutil -m cp gs://208220756/Titles "/content"') #Titles

name = 'Doc_Len'
with open(Path('/content') / f'{name}', 'rb') as f:
             DL = pickle.loads(f.read())

name = 'Titles'
with open(Path('/content') / f'{name}', 'rb') as f:
             Titles = pickle.loads(f.read())


class MyFlaskApp(Flask):
    index_body = InvertedIndex.read_index("/content","index_body")
    index_title = InvertedIndex.read_index("/content","index_title")
    index_anchor = InvertedIndex.read_index("/content","index_anchor")
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    index_body =  InvertedIndex.read_index("/content","index_body")
    index_title = InvertedIndex.read_index("/content","index_title")
    lst_query = [token.group() for token in RE_WORD.finditer(query.lower())]
    filtered_query = [token for token in lst_query if token not in list(all_stopwords)]
    body_scores = get_relevant(filtered_query,index_body)
    title_scores = get_relevant(filtered_query,index_title)
    merged = merge_results(body_scores,title_scores)
    for k,v in merged:
      res.append( (k,Titles[k]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    index_body =  InvertedIndex.read_index("/content","index_body")
    lst_query = [token.group() for token in RE_WORD.finditer(query.lower())]
    terms = [element for element in index_body.posting_locs]
    filtered_query = [token for token in lst_query if token in terms and token not in list(all_stopwords)]

    cs= cosine_similarity(filtered_query,index_body,DL)
    top_relevant= get_top(cs,100)
    for doc_id,score in top_relevant:
      res.append((doc_id,Titles[doc_id]))

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    index_title = InvertedIndex.read_index("/content","index_title")
    lst_query = [token.group() for token in RE_WORD.finditer(query.lower())]
    dic={}
    for token in lst_query:
      if token in all_stopwords:
        continue
      for doc_id,tf in read_posting_list(index_title,token):
        dic[doc_id]= dic.get(doc_id,0)+tf
    res = [ ( k , Titles[k] ) for k, v in sorted(dic.items(), key=lambda item: item[1],reverse=True)]
  

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    index_anchor = InvertedIndex.read_index("/content","index_anchor")
    lst_query = [token.group() for token in RE_WORD.finditer(query.lower())]
    dic={}
    for token in lst_query:
      if token in all_stopwords:
        continue
      for doc_id,tf in read_posting_list(index_anchor,token):
        dic[doc_id]= dic.get(doc_id,0)+1
    res = [ ( k , Titles[k] ) for k, v in sorted(dic.items(), key=lambda item: item[1],reverse=True)]
  
    # END SOLUTION
    return jsonify(res)
@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.`
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
  
    df = pd.read_csv('gs://208220756/pr/part-00000-95e708d4-66ea-489e-a98d-5dc757eceb43-c000.csv.gz', encoding='utf-8',compression='gzip')
    df.columns = ['id','pagerank']
    mask = df['id'].isin(wiki_ids)
    df = df.loc[mask]
    res = df['pagerank'].tolist() 
    

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
