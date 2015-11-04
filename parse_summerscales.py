import os 
import pdb 

import numpy as np  

import nltk 

import BeautifulSoup as bs 

def get_tokens_and_lbls(annotated_data_path="summerscales-annotated-abstracts"): 
    pmids, docs, lbls = [], [], []

    for f in _get_xml_file_names(annotated_data_path):
        print f
        soup = bs.BeautifulSoup(open(f).read())
        pmid = soup.find("abstract")['id']
        pmids.append(pmid)

        sentences = list(soup.findAll("s"))
        ordered_sentences = []
        sentence_ids = [int(sent["id"]) for sent in sentences]
        d = dict(zip(sentence_ids, sentences))
        for s_id in sorted(d.keys()):
            ordered_sentences.append(d[s_id])

        doc_sentence_tokens, doc_sentence_lbls = [], []

        for sent in ordered_sentences:
            cleaned_sent, group_tokenized, sent_tokenized = [None]*3 # for sanity

            group_str_in_sent = sent.find("group")
            if group_str_in_sent is None: 
                sent_tokenized = nltk.word_tokenize(sent.text)
                sent_lbls = [-1 for _ in sent_tokenized]
            else: 
                # then this contains at least one group mention!
                sent_lbls = []
                group_text = sent.group.text

                # now we have to deal with a small annoyance;
                # the 'short' tags end up removing spaces
                # from the text in the group tags, e.g.:
                #   u'home  basedmedication  review'
                # instead of 
                #   u'home  based medication  review'
                # here we fix that.
                short_name = sent.short
                if short_name is not None: 
                    group_text = group_text.replace(short_name.text, " %s " % short_name.text)
                    # also need to fix it up in the whole sentence!
                    cleaned_sent = sent.text.replace(short_name.text, " %s " % short_name.text)

                else: 
                    cleaned_sent = sent.text 

                cleaned_sent = cleaned_sent.replace(group_text, " %s " % group_text)

                group_tokenized = nltk.word_tokenize(group_text)
                sent_tokenized  = nltk.word_tokenize(cleaned_sent)
                
                # initialize label vector for the current sentence to -1s
                sent_lbls = [-1 for _ in sent_tokenized]
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

        docs.extend(doc_sentence_tokens)
        lbls.extend(doc_sentence_lbls)

    return pmids, docs, lbls
        #docs.append(all_sentences)


def _get_xml_file_names(dir_):
    return [os.path.join(dir_, f) 
        for f in os.listdir(dir_) if f.endswith(".xml")]