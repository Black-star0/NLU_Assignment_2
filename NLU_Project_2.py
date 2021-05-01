import conll
from conll import *
import spacy
import pandas as pd
from spacy2conll import spacytoconll
from spacy.tokens import Doc, Span
from sklearn.metrics import classification_report
from collections import Counter

#Trento University - Master of Artificial Intelligence Systems - Second Assignment of Natural Language Understanding Course 
#Student Name: Alessandro Pini

#The assignment is implemented in Python using the spaCy, Sklearn library and Pandas library. All of them have to be installed in order to run correctly this script.

test_path = 'DATA/test.txt'
#train_path = 'DATA/train.txt'
#dev_path = 'DATA/dev.txt'

#the dataset used is “conll_2003”, useful for the statistical evaluation part, it can be download here: https://github.com/esrel/NLU.Lab.2021/tree/master/src 
#I already put it and etracted it in the folder's project.
# Read dataset

#For a more accurate results it's possible to evaluate all data that include train data + test data + dev data. 
# In this case I decided to test my functions only on test.txt. 

txt = read_corpus_conll(test_path, fs=' ')
#txt.extend(read_corpus_conll(train_path, fs=' '))
#txt.extend(read_corpus_conll(dev_path, fs=' '))

# I removed from my dataset all -DOCSTART- sentences that are usually used to separate different documents from each other.

txt = list(filter(lambda s: s[0][0] != '-DOCSTART-', txt)) 

#Here's a parsing with spacy
nlp = spacy.load('en_core_web_sm')
doc = Doc(nlp.vocab, [w[0] for s in txt for w in s])


# Custom sentence split to allineate the spaCy results to CoNLL
# To make comparable spacy results with ConLL I split the sentence paying attention to set y == 0. is sent_start serve as a boundary margin, 
# so the variable is set True only for the first word of the sentence

i = 0
for s in txt:
    for y, word in enumerate(s):
            doc[i].is_sent_start = (y == 0)
            i += 1

for name, processed in nlp.pipeline:
    doc = processed(doc)

#SpaCy uses a wider set of labels w.r.t CoNLL. So, in order to evaluate SpaCy's parser performance, 
# I converted spacy tag to the CoNLL format. A dictionary containing a label map is declared below:

map = {
        'PERSON': 'PER',
        'ORG': 'ORG',
        'LOC': 'LOC',
        'GPE': 'LOC',
        'NORP': 'MISC',
        'PRODUCT': 'MISC',
        'EVENT': 'MISC',
        'FAC': 'MISC',
        'WORK_OF_ART': 'MISC',
        'LAW': 'MISC',
        'LANGUAGE': 'MISC'
    }

#Here the function used to perform the convertion
# Convert spacy tags to conll

def spacytoconll(iob_tag, type):

    if not type in map:

        return 'O'
    else:
        return iob_tag + '-' + map[type]

#------------------------------------------TASK 1-------------------------------------------------

#Evaluate spaCy NER on CoNLL 2003 data (provided)

#report token-level performance (per class and total)
# - accuracy of correctly recognizing all tokens that belong to named entities (i.e. tag-level accuracy)
#report CoNLL chunk-level performance (per class and total);
# - precision, recall, f-measure of correctly recognizing all the named entities in a chunk per class and total       


def eval(doc):

    #Before computing the actual evaluation using the conll.evaluate function privided by prof, I did a pre-processing in order to align the right format
    hyp = [[(word.text, spacytoconll(word.ent_iob_, word.ent_type_)) 
        for word in s] 
            for s in doc.sents]

    ref = [[(word[0], word[3]) 
        for word in s] 
            for s in txt]

    #After the pre-processing step I computer the performance at Token-level and at Chuck-Level using the evaluate function defined in conll.
    #For the first case I used the classification_report included in scikit_learn library. In this case that function took as input entity and iob type. 
    # conll.evaluate take otherwise hyp and ref

    # performance at Token-level 
    t_perf = classification_report([w[1] for s in ref for w in s], [w[1] for s in hyp for w in s])

    # performance at Chunk-level 
    c_perf = conll.evaluate(ref, hyp)
    c_perf = pd.DataFrame.from_dict(c_perf, orient='index')

    return t_perf, c_perf


# Compute performances
t_perf, c_perf = eval(doc)
print('Performance at Token-Level:\n', t_perf)
print('\n')
print('\n')
print('Performance at Chuck-Level:\n', c_perf)

'''
#Output:
Performance at Token-Level:
               precision    recall  f1-score   support

       B-LOC       0.79      0.68      0.73      1668
      B-MISC       0.70      0.55      0.62       702
       B-ORG       0.51      0.29      0.37      1661
       B-PER       0.78      0.61      0.68      1617
       I-LOC       0.59      0.60      0.59       257
      I-MISC       0.42      0.40      0.41       216
       I-ORG       0.46      0.50      0.48       835
       I-PER       0.83      0.74      0.78      1156
           O       0.95      0.98      0.96     38323

    accuracy                           0.91     46435
   macro avg       0.67      0.59      0.63     46435
weighted avg       0.90      0.91      0.90     46435


Performance at Chuck-Level:
               p         r         f     s
MISC   0.693989  0.542735  0.609113   702
LOC    0.776706  0.675659  0.722668  1668
PER    0.757143  0.589981  0.663191  1617
ORG    0.455128  0.256472  0.328071  1661
total  0.688275  0.511331  0.586753  5648

'''

#----------------------------------------TASK 2----------------------------------------------------

#Grouping of Entities. Write a function to group recognized named entities using noun_chunks method of spaCy. 
# - Analyze the groups in terms of most frequent combinations (i.e. NER types that go together).


def get_sent(txt):

  sents = []
  for s in txt:
    list_tk = []
    for tk in s:
      string_t = tk[0].split(' ')[0]
      list_tk.append(string_t)
    sents.append(' '.join(list_tk))
  return sents

def get_ne(list_s):

  sent_groups = []
  for s in list_s:
    doc = nlp(s)
    #set boundaries of the noun_chucks
    noun_chunk = []
    for c in doc.noun_chunks:
      if c.ents:
        noun_chunk.append(c.start)
        noun_chunk.append(c.end)
    groups = []
    if not noun_chunk:
      for ne in doc.ents:
        groups.append(sorted(list(dict.fromkeys([tk.ent_type_ for tk in ne]))))
    else:
      noun_chunk_i = 0
      for ne in doc.ents:
        group = []
        first = ne.start
        if first < noun_chunk[min(noun_chunk_i+1, len(noun_chunk)-1)] and first >= noun_chunk[noun_chunk_i]:
          for tk in doc[noun_chunk[noun_chunk_i]:noun_chunk[noun_chunk_i+1]]:
            if tk.ent_type_ not in group and tk.ent_type_ != "": 
              group.append(tk.ent_type_)
          groups.append(sorted(group))
          noun_chunk_i = min(noun_chunk_i+2, len(noun_chunk)-1)
        elif first >= noun_chunk[max(0,noun_chunk_i-1)] or first >= noun_chunk[noun_chunk_i] and first < noun_chunk[noun_chunk_i]:
          groups.append(sorted(list(dict.fromkeys([tk.ent_type_ for tk in ne]))))
    sent_groups.append(groups)
  return sent_groups

corp_list = get_sent(txt)

sent_group_ne = get_ne(corp_list)


group_counts = Counter()
for s in sent_group_ne:
  for group in s:
    group_counts.update([' '.join(group)])
group_counts.most_common()

print(group_counts)
print('\n')
print('\n')

#---------------------------------------TASK 3--------------------------------------------

#One of the possible post-processing steps is to fix segmentation errors. 
# - Write a function that extends the entity span to cover the full noun-compounds. 
# - Make use of compound dependency relation.

#Here's I did a simple join between tokens that are adjacent to at least a named entity in a compund relation with it. 
# In this way it's possible to extend the entity span to cover the full noun compound 

#I used the built-in function doc.set_ents of Spacy to implement this algorithm  passing all the named entities catched in the Doc object.
def merge_compounds(doc):

  ne1 = []
#when a token is in compound relation with multiple entities
  dependency_compound = set() 
  for t in doc.ents:
    s = t.start
    e = t.end
    if t.label != "":
      catch = False
      for tok in t:
        if (doc[s-1].dep_ == "compound" and (doc[max(s-1, 0)].head == tok and 
            doc[max(s-1, 0)].ent_type_ == "")
          or
          (doc[max(s-1, 0)] == tok.head and tok.dep_ == "compound" and 
           doc[max(s-1, 0)].ent_type_ == "")):
          if s-1 not in dependency_compound:
            catch = True
            ne1.append(Span(doc, s-1, e, t.label_))
            dependency_compound.add(s-1)
            break

        elif ((doc[min(e, len(doc)-1)].head == tok and doc[min(e, len(doc)-1)].dep_ == "compound" and 
                doc[min(e, len(doc)-1)].ent_type_ == "") or
            (doc[min(e, len(doc)-1)] == tok.head and tok.dep_ == "compound" and 
            doc[min(e, len(doc)-1)].ent_type_ == "")):
          catch = True
          ne1.append(Span(doc, s, e+1, t.label_))
          dependency_compound.add(e)
          break

      if not catch:
        ne1.append(Span(doc, s, e, t.label_))
  doc.set_ents(ne1)
  return doc

#I the end I evaluated again with the function already used before the spacy model integrated with the dependency_compaund. 

# Re-Compute performances
new_t_perf, new_c_perf = eval(merge_compounds(doc))

print('Performance at Token-Level:\n', new_t_perf)
print('\n')
print('\n')
print('Performance at Chunk-Level:\n', new_c_perf)

#Output
'''
Performance at Token-Level:
               precision    recall  f1-score   support

       B-LOC       0.77      0.67      0.72      1668 
      B-MISC       0.70      0.55      0.62       702
       B-ORG       0.50      0.28      0.36      1661
       B-PER       0.67      0.53      0.59      1617
       I-LOC       0.39      0.60      0.47       257
      I-MISC       0.33      0.40      0.36       216
       I-ORG       0.40      0.51      0.45       835
       I-PER       0.70      0.75      0.72      1156
           O       0.95      0.97      0.96     38323

    accuracy                           0.89     46435
   macro avg       0.60      0.58      0.58     46435
weighted avg       0.89      0.89      0.89     46435


Performance at Chunk-Level:
               p         r         f     s
MISC   0.632058  0.494302  0.554756   702
LOC    0.714680  0.621703  0.664957  1668
PER    0.647619  0.504638  0.567258  1617
ORG    0.370726  0.208910  0.267231  1661
total  0.607007  0.450956  0.517473  5648
'''

#Results shows that does not seem beneficial in the evaluation.

#----------------------------------------------END----------------------------------------------
