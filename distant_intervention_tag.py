import pdb 
import string 

import cochranenlp
from cochranenlp.readers.biviewer import BiViewer
from cochranenlp.textprocessing.drugbank import Drugbank

import nltk
from nltk.corpus import stopwords 

from noaho import NoAho

import fuzzywuzzy

drugbank = Drugbank()
stopwords = stopwords.words('english')
# hand-crafted and intended for targeting interventions!
# some of these are just words that are likely to be shared
# between cdsr text and abstract, even though not describing
# interventions specifically
stopwords.extend(["either", "time", "patients", "or"])

def annotate():
    bviewer = BiViewer()

    # start with non-drug trials that us the 'vs' 
    # construction, e.g., "exercise vs diet"
    # note that this explicitly does not consider
    # drug trials
    versus_studies = _find_vs(bviewer)
    tag_versus_studies(versus_studies)

def _tag_drugs(study):
    intervention_text = _iv_for_study(study)
    abstract = _abstract_for_study(study)

    interventions_tokens = nltk.word_tokenize(intervention_text)
    abstract_tokens = nltk.word_tokenize(abstract)

    return get_drugs(abstract_tokens, interventions_tokens)

def _match_placebo(abstract_tokens, intervention_tokens):
    abstract_placebo = _find_token(abstract_tokens, "placebo")
    cdsr_placebo = _find_token(intervention_tokens, "placebo")
    # if they both mention placebo, tag all mentions in the 
    # abstract
    if len(abstract_placebo) > 0 and len(cdsr_placebo) > 0: 
        return abstract_placebo
    return -1

def get_drugs(abstract_tokens, interventions_tokens, add_placebo=True):
    abstract_drugs = drugbank._find_longest_token_matches(abstract_tokens)
    cdsr_drugs = drugbank._find_longest_token_matches(interventions_tokens)

    shared = set([dr[0] for dr in cdsr_drugs]).intersection(
                set([dr[0] for dr in abstract_drugs]))
    matched_indices = []
    for dr in abstract_drugs:
        drug_name, start, end = dr
        if drug_name in shared:
            matched_indices.append(dr)

    if add_placebo:
        matched_placebo = _match_placebo(abstract_tokens, interventions_tokens)
        if matched_placebo != -1:
            matched_indices.extend(matched_placebo)

    return matched_indices

def _find_token(seq, t):
    indices = []
    for i, t_j in enumerate(seq):
        if t_j.lower() == t:
            indices.append((i, i, t))
    return indices

def _iv_for_study(study):
    return study.cochrane['CHARACTERISTICS']['CHAR_INTERVENTIONS']

def _abstract_for_study(study):
    return study.pubmed["abstract"]

def _is_drug_trial(study):
    # do both the iv text and abstract mention a drug??
    intervention_text = _iv_for_study(study)
    abstract_text = study.pubmed['abstract']

    is_drug_trial = drugbank.contains_drug(intervention_text.lower()) and \
                    drugbank.contains_drug(abstract_text.lower())

    return is_drug_trial

def _find_vs(bviewer, include_drug_trials=False):
    studies = []
    for study in bviewer:
        if _iv_for_study(study) is None or _abstract_for_study(study) == "":
            pass 
        elif "vs" in _iv_for_study(study).split(" "):
            if not _is_drug_trial(study) or include_drug_trials:
                studies.append(study)
    return studies

def _is_num(t):
    try: 
        float(t)
        return True
    except:
        return False 

def _clean_tokens(tokens):
    # remove stopwords and punctuation 
    cleaned = []
    for t in tokens:
        if t not in string.punctuation and t not in stopwords and \
           len(t) > 1 and not _is_num(t):
            cleaned.append(t)
    return cleaned 
    
def _match_and_tag(iv_tokens, abstract_tokens, top_k=1):
    # match longest substring
    matched, spans = longest_common_substrings(
                        abstract_tokens, 
                        iv_tokens, return_spans=True)
    
    matched_lens = [len(_clean_tokens(match)) for match in matched]
    if len(matched_lens) == 0:
        max_lens = []
    else:
        max_lens = [len_ for len_ in sorted(matched_lens, reverse=True)[:top_k] 
                        if len_ > 0]

    if len(max_lens) == 0:
        # no matches.
        return []
    else:
        longest_indices = [idx for idx, x in enumerate(matched_lens) if x in max_lens] #matched_lens.index(max_len)
        best_matches, tagged_spans = [], []
        for idx in longest_indices:
            tagged_spans.append((" ".join(matched[idx]), spans[idx][0], spans[idx][1]))
          
    return tagged_spans


def _tag_general_nondrug(study):
    '''
    @TODO there is a lot of overlap with _tag_vs 
    to be refactored!
    '''
    intervention_text = _iv_for_study(study)
    abstract = _abstract_for_study(study)

    iv_tokens = nltk.word_tokenize(intervention_text.lower())
    abstract_tokens = nltk.word_tokenize(abstract.lower())
    
    tagged_spans = _match_and_tag(iv_tokens, abstract_tokens, top_k=2)

    return tagged_spans

def _tag_vs(study):
    intervention_text = _iv_for_study(study)
    abstract = _abstract_for_study(study)
    assert("vs" in intervention_text)

    interventions = intervention_text.split("vs")
    for iv in interventions:
        iv_tokens = nltk.word_tokenize(iv.lower())
        abstract_tokens = nltk.word_tokenize(abstract.lower())
        # match longest substring
        matched, spans = longest_common_substrings(
                            abstract_tokens, 
                            iv_tokens, return_spans=True)
        
        matched_lens = [len(_clean_tokens(match)) for match in matched]
        if len(matched_lens) == 0:
            max_len = 0
        else:
            max_len = max(matched_lens)

        if max_len == 0:
            #return -1 
            return []
        else:
            longest_indices = [idx for idx, x in enumerate(matched_lens) if x == max_len] #matched_lens.index(max_len)
            best_matches, tagged_spans = [], []
            for idx in longest_indices:
                if matched[idx] not in best_matches:
                    best_matches.append(matched[idx])
                    

                tagged_spans.append((" ".join(matched[idx]), spans[idx][0], spans[idx][1]))
                
    return tagged_spans

def _tag_tokens(tokens_list, t):
    indices = []
    for t_j in tokens_list:
        pass

def tag_versus_studies(versus_studies):
    pos_indices = []
    for study in versus_studies:
        study_pos_indices = _tag_vs(_iv_for_study(study), 
                                _abstract_for_study(study))
        if study_pos_indices != -1:
            pass 

def ds_interventions_abstracts(bviewer, num_studies=None):
    tagged_pmids, tagged_abstracts, tagged_iv_texts, abstracts = [], [], [], []
    count = 0
    for study in bviewer:
        if count > num_studies:
            break

        cur_pmid = study.pubmed['pmid']
        '''
        abstract = study.pubmed['abstract']
        intervention_cdsr = _iv_for_study(study)
        '''
        ## first, is the 
        intervention_text = _iv_for_study(study)
        abstract_text = _abstract_for_study(study)
        if intervention_text is None or abstract_text == "":
            pass
        else: 
            tagged = None
            if "vs" in intervention_text.split(" ") and not _is_drug_trial(study):
                #tagged_abstracts.append(_tag_vs(study))
                tagged = _tag_vs(study)
            elif not _is_drug_trial(study):
                # this is non-drug trial and 
                tagged = _tag_general_nondrug(study)
            else:
                # drug study!
                tagged = _tag_drugs(study)

            
            if len(tagged) > 0:
                tagged_abstracts.append(tagged)
                abstracts.append(abstract_text)
                tagged_pmids.append(cur_pmid)
                tagged_iv_texts.append(intervention_text)
                count += 1

    pdb.set_trace()
    return tagged_pmids, tagged_abstracts

def longest_common_substrings(s1, s2, return_spans=False):
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    longest_list = []
    spans = []
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] >= longest:
                    longest = m[x][y]
                    x_longest = x
                    longest_list.append(s1[x_longest - longest: x_longest])
                    spans.append((x_longest - longest, x_longest))
            else:
                m[x][y] = 0

    if return_spans:
        return longest_list, spans
    return longest_list#s1[x_longest - longest: x_longest]

