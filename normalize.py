__author__ = 'zhangye'
import re

from geniatagger import *

import drugbank


def normalize(input,drug):
    #replace drug name by "DRUG"
    input = drug.sub(input)

    p1 = re.compile("\d+(\.*\d+)*\s*-\s*\d+(\.*\d+)*")
    input = p1.sub(' MEASURE_RANGE ',input)

#match measure   eg. mug/d
    before_slash = ["g","mg","kg","ng","ug",'mug','microg','mcg']
    after_slash = ["day","dl","d","h","ml",'kg']
    before = '|'.join(before_slash)
    after = '|'.join(after_slash)
    p2 = re.compile('('+before+')'+'/'+'('+after+')',re.IGNORECASE)
    input = p2.sub(" MEASURE_UNIT ",input)

    #match equal relationship
    p3 = re.compile("n\s*=\s*\d+")
    input = p3.sub(" EQUAL_REL ",input)

    #match measure  250 mg
    p4 = re.compile("\d+\.*\d+\s*"+"("+before+")")
    input = p4.sub(" MEASURE_UNIT ",input)

    #match ratio
    p8 = re.compile("\d+\s*:\s*\d+")
    input = p8.sub(" RATIO_POP ",input)

    #replace percentage
    p7 = re.compile("\d+\.\d+%")
    input = p7.sub(" PERCENTAGE ",input)

    #replace all float numbers
    p6 = re.compile(r"\b\d+\.\d+\b")
    input = p6.sub("FLOAT",input)

    #replace all integers
    p5 = re.compile(r"\b\d+\b")
    input = p5.sub("INTEGER", input)
    return  input
def main():
    drug = drugbank.Drugbank()
    tagger = GeniaTagger('/Users/zhangye/Documents/Study/UTAustin/study/causal_Statement/geniatagger-3.0.1/geniatagger')
    str = 'After drug washout and a 1- to 3-week antipsychotic-free period,\
     patients were randomized to treatment with clozapine (n = 12) or olanzapine (n = 13).'
    print normalize(str,drug)

    #print SNOMEDCT.search('In total, 316 subjects were randomized (159 to colesevelam hydrochloride, 3.75 g/d, and 157 to matching placebo).')
    #for c in SNOMEDCT.CORE_problem_list(): print c
if __name__ == "__main__":
    main()