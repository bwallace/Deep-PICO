import os
import pdb
import string
import matplotlib
matplotlib.use('Agg')

import numpy as np

import nltk
import BeautifulSoup as bs
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from geniatagger import *
import sys
#nltk.data.path.append('/work/03186/ericr/nltk_data/')

def _just_the_txt(s):
    return " ".join(s.findAll(text=True)).strip()

def get_tokens_and_lbls(annotated_data_path="summerscales-annotated-abstracts",
                        start_and_stop_tokens=False, make_pmids_dict=False, classify=False, sen=False,
                        tagger_path='', parse_for_cnn=False, window_size=5):

    #tagger = GeniaTagger('/home1/03186/ericr/code/geniatagger-3.0.1/geniatagger')
    tagger = GeniaTagger('/Users/ericrincon/Downloads/geniatagger-3.0.2/geniatagger')
    stop_token =  "STOPSTOPSTOP"
    start_token = "STARTSTARTSTART"
    sentence_docs = []
    pmids, docs, lbls = [], [], []
    pmids_dict = {}
    sentence_labels = []
    doc_labels = []
    file_names = _get_xml_file_names(annotated_data_path)
    g_i = 0
    groups_per_abstract_dict = {}
    file_names_clean = []
    group_counts_sents = {}
    xml_sentences = []
    groups_map = {}

    abstracts_as_tokens = []
    abstract_output_labels = []
    pos_tag_dict = {}
    chunk_tag_dict = {}
    ne_tag_dict = {}
    dicts = {}

    abstract_as_window= []
    for f in file_names:
        print(f)
        file_names_clean.append(f.split('/')[-1])
        sen_labels = []
        groups_per_abstract = []
        doc_sentences = []
        soup = bs.BeautifulSoup(open(f).read())
        pmid = soup.find("abstract")['id']

        sentences = list(soup.findAll("s"))

        ordered_sentences = []
        ordered_sentences_clean = []
        sentence_ids = [int(sent["id"]) for sent in sentences]
        d = dict(zip(sentence_ids, sentences))
        group_counts = []
        groups_abs = []
        abstract_ids = []
        abstract_labels = []
        abstract_words = []
        tagged_abstract = []

        abstract_groups = []
        abstract_outcomes = []
        abstract_conditions = []
        mentions = []
        groups_dict = {}
        groups_dicts = []
        for i, s_id in enumerate(sorted(d.keys())):
            cur_s = d[s_id]
            cleaned_sent = _just_the_txt(cur_s)
            cleaned_sent_words = cleaned_sent.split()

            # including tags
            ordered_sentences.append(cur_s)

            groups = cur_s.findAll('group')

            sentence_ids = []
            sentence_groups = []

            group_words = set([])

            for group in groups:
                if group.short is not None:
                    for group_word in group.short.text.split():
                        group_words.add(group_word)
            abstract_groups.append(group_words)



            for group in groups:
                if int(group['id'] not in groups_dict):
                    if len(group.findAll('short')) > 0:
                        groups_dict[int(group['id'])] =  group.findAll('short')[0].text
                  #  groups_dict[int(group['id'])] =


                if int(group['id']) not in sentence_ids:
                    sentence_ids.append(int(group['id']))
                if int(group['id']) not in abstract_ids:

                    abstract_ids.append(int(group['id']))

                    if len(group.findAll('short')) > 0:
                        for group_tag in group.findAll('short'):
                            token_text = nltk.word_tokenize(group_tag.text)

                            for token in token_text:
                                if token not in groups_abs:

                                    sentence_groups.append(token)
                                    groups_abs.append(token)
                    elif group.text is not None:
                        token_text = nltk.word_tokenize(group.text)

                        for token in token_text:
                            if token not in groups_abs:

                                sentence_groups.append(token)
                                groups_abs.append(token)
                    """
                    if group.text is not None:
                        groups_abs.append(group.text)
                        sentence_groups.append(group.text)
                    elif group.find('short') is not None:
                        for group_tag in group.findAll('short'):
                            groups_abs.append(group_tag.text)
                            sentence_groups.append(group.text)
                    """
            tagged_sentence = tagger.parse(cleaned_sent)
            cleaned_tags = []
            for tag in tagged_sentence:
                tags = (tag[2], tag[3], tag[4])
                cleaned_tags.append(tags)

            tagged_abstract.extend(cleaned_tags)
            group_counts.append(len(sentence_ids))
            # stripped of tags; just the text
            ordered_sentences_clean.append(cleaned_sent)
            tokenized_sentence = nltk.word_tokenize(cleaned_sent)

            for word in tokenized_sentence:

                abstract_words.append(word)
                if word in sentence_groups:
                    abstract_labels.append(1)
                else:
                    abstract_labels.append(0)
        if len(groups_abs) == 0:
            continue
        groups_per_abstract_dict.update({f.split('/')[-1]: len(abstract_ids)})
        group_counts_sents.update({f.split('/')[-1]: group_counts})
        groups_map[pmid] = groups_abs



        """
        if start_and_stop_tokens:
            doc_sentence_tokens, doc_sentence_lbls = [[start_token]], [[-1]]
        else:
            doc_sentence_tokens, doc_sentence_lbls = [], []

        for sent_idx, sent in enumerate(ordered_sentences):
            #cleaned_sent, group_tokenized, sent_tokenized = [None]*3 # for sanity
            cleaned_sent = ordered_sentences_clean[sent_idx]
            #sent_tokenized = nltk.word_tokenize(cleaned_sent)
            #words = nltk.word_tokenize(str(sent))
            words = nltk.word_tokenize(str(cleaned_sent))

            #doc_sentences.append(string.join(words))
            # initialize label vector for the current sentence to -1s
           # sent_lbls = [-1 for _ in sent_tokenized]

            group_strs_in_sent = sent.findAll("group")

            group_texts = []

            for group_str in group_strs_in_sent:
                group_text = _just_the_txt(group_str) #group_str.text
                print(group_text)
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
            if 1 in sent_lbls:
                sen_labels.append(1)
            else:
                sen_labels.append(-1)
            doc_sentence_tokens.append(sent_tokenized)

           # abstract_words.extend(words)

            for word in words:
                contains = False

                for group_word in groups_abs:
                    if word in group_word:
                        contains = True
                        break

                if contains:
                    abstract_labels.append(1)
                else:
                    abstract_labels.append(0)
        """
        abstracts_as_tokens.append(abstract_words)
        abstract_output_labels.append(abstract_labels)
        pmids_dict[pmid] = (abstract_words, abstract_labels, tagged_abstract, groups_dict, abstract_groups)

        """
        doc_labels.append(sen_labels)
        if start_and_stop_tokens:
            doc_sentence_lbls.append([-1])
            doc_sentence_tokens.append([stop_token])
            #pmids.append([pmid])


        if make_pmids_dict:
            pmids_dict[pmid] = (doc_sentence_tokens, doc_sentence_lbls)



        docs.extend(doc_sentence_tokens)
        lbls.extend(doc_sentence_lbls)
        pmids.extend([pmid]*len(doc_sentence_lbls))
        sentence_docs.append(doc_sentences)
        """

    n, bins, patches = plt.hist(groups_per_abstract_dict.values(), 50, normed=0, facecolor='green', alpha=0.75)
    plt.savefig('histo.png')

    ## just use sklearn to get indices for vocab
    """
    V = []
    for d in docs:
        V.extend(d)

    # will probably blow up if you introduce more data...
    vocab = list(set(V))
    v = CountVectorizer(ngram_range=(1,1), vocabulary=vocab, binary=True)
    texts = [" ".join(doc) for doc in docs]
    v.fit([" ".join(doc) for doc in docs])
    """
    V = []
    for abstract in abstracts_as_tokens:
        V.extend(abstract)

    vocab = list(set(V))
    v = CountVectorizer(ngram_range=(1,1), vocabulary=vocab, binary=True)

    texts = [" ".join(abstract) for abstract in abstracts_as_tokens]
    v.fit([" ".join(abstract) for abstract in abstracts_as_tokens])

    """
    if classify:
        print group_counts_sents
        return sentence_docs, doc_labels, file_names_clean, groups_per_abstract_dict, group_counts_sents
    if sen:
        return pmids_dict, pmids, doc_sentence_tokens, doc_sentence_lbls, v, groups_map

    if make_pmids_dict:
        return pmids_dict, pmids, docs, lbls, v, groups_map
    else:
        return pmids, docs, lbls, v
    """

    one_hot = OneHotEncoder()
    one_hot_values = []

    for value in v.vocabulary_.values():
        one_hot_values.append([value])
    one_hot.fit(one_hot_values)
    dicts['pos'] = pos_tag_dict
    dicts['ne'] = ne_tag_dict
    dicts['chunk'] = chunk_tag_dict

    return pmids_dict, pmids, abstracts_as_tokens, abstract_output_labels, v, groups_map, one_hot, dicts

def _get_xml_file_names(dir_):
    return [os.path.join(dir_, f)
        for f in os.listdir(dir_) if f.endswith(".xml")]