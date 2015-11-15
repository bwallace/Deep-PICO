import os 
import pdb 

import numpy as np  

import nltk 
import BeautifulSoup as bs 

from sklearn.feature_extraction.text import CountVectorizer


def _just_the_txt(s):
    return " ".join(s.findAll(text=True)).strip()

##
# fix this. a potential solution!!!
# http://stackoverflow.com/questions/2957013/beautifulsoup-just-get-inside-of-a-tag-no-matter-how-many-enclosing-tags-there
def get_tokens_and_lbls(annotated_data_path="summerscales-annotated-abstracts", 
                            start_and_stop_tokens=True): 
    
    stop_token =  "STOPSTOPSTOP"
    start_token = "STARTSTARTSTART"

    pmids, docs, lbls = [], [], []
    
    for f in _get_xml_file_names(annotated_data_path):
        soup = bs.BeautifulSoup(open(f).read())
        pmid = soup.find("abstract")['id']
        sentences = list(soup.findAll("s"))
        
        ordered_sentences = []
        ordered_sentences_clean = []
        sentence_ids = [int(sent["id"]) for sent in sentences]
        d = dict(zip(sentence_ids, sentences))

        for s_id in sorted(d.keys()):
            cur_s = d[s_id]
            # including tags
            ordered_sentences.append(cur_s)
            # stripped of tags; just the text
            ordered_sentences_clean.append(_just_the_txt(cur_s))

        if start_and_stop_tokens:
            doc_sentence_tokens, doc_sentence_lbls = [[start_token]], [[-1]]
        else:
            doc_sentence_tokens, doc_sentence_lbls = [], []

        for sent_idx, sent in enumerate(ordered_sentences):
            cleaned_sent, group_tokenized, sent_tokenized = [None]*3 # for sanity
            cleaned_sent = ordered_sentences_clean[sent_idx]
            sent_tokenized = nltk.word_tokenize(cleaned_sent)
                   
            # initialize label vector for the current sentence to -1s
            sent_lbls = [-1 for _ in sent_tokenized]

            group_strs_in_sent = sent.findAll("group")
            group_texts = []
            for group_str in group_strs_in_sent:
                group_text = _just_the_txt(group_str) #group_str.text

                group_tokenized = nltk.word_tokenize(group_text)
                
                # and now flip on labels corresponding to the tokens in the
                # group span we read out
                try:
                    start_index = sent_tokenized.index(group_tokenized[0])
                except:
                    pdb.set_trace()
                group_len = len(group_tokenized)
                sent_lbls[start_index:start_index+group_len] = [1]*group_len
            
            doc_sentence_lbls.append(sent_lbls)
            doc_sentence_tokens.append(sent_tokenized)



        if start_and_stop_tokens:
            doc_sentence_lbls.append([-1])
            doc_sentence_tokens.append([stop_token])
            #pmids.append([pmid])

        docs.extend(doc_sentence_tokens)
        lbls.extend(doc_sentence_lbls)
        #b.set_trace()
        pmids.extend([pmid]*len(doc_sentence_lbls))

            
    
    ## just use sklearn to get indices for vocab
    V = []
    for d in docs:
        V.extend(d)

    # will probably blow up if you introduce more data... 
    vocab = list(set(V))
    v = CountVectorizer(ngram_range=(1,1), vocabulary=vocab, binary=True)
    texts = [" ".join(doc) for doc in docs]
    v.fit([" ".join(doc) for doc in docs])

    return pmids, docs, lbls, v




def _get_xml_file_names(dir_):
    return [os.path.join(dir_, f) 
        for f in os.listdir(dir_) if f.endswith(".xml")]