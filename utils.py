import re
import string
import collections
from collections import OrderedDict
import matplotlib as mpl
from statistics import mean
import matplotlib.pyplot as plt


def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def get_raw_scores(dataset, preds):
  # get max f1 scores for each question
  exact_scores = {}
  f1_scores = {}
  for article in dataset['data']:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  # computes total f1 and exact scores for the model
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])

def computes_f1(dataset, f1_raw):
  # returns a list of dictionaries with length of correct answer and f1 score
  f1_len = []

  for article in dataset['data']:
      for p in article['paragraphs']:
        for qa in p['qas']:
          qid = qa['id']
          f1_score = f1_raw[qid]
          gold_len = [len(a['text']) for a in qa['answers']
                          if normalize_answer(a['text'])]
          f1_len.append({'len': gold_len[0], 'f1': f1_score})
                  
  return f1_len

def compute_avg_f1(f1_len):
  # computes the average f1 score per answer length

  # using sorted and lambda to sort list by len
  sorted_f1 = sorted(f1_len, key = lambda i: i['len'])

  f1_avg_len = []
  ans_len = []
  len = sorted_f1[0]['len']
  f1_tot = sorted_f1[0]['f1']
  tot = 1
  
  for item in sorted_f1:
    if item['len'] == len:
      # increment counts
      f1_tot += item['f1']
      tot += 1
    elif item['len'] > len:
      # compute avg
      f1_avg = f1_tot / tot
      f1_avg_len.append(f1_avg)
      ans_len.append(len)
      # reset
      len = item['len']
      f1_tot = item['f1']
      tot = 1
  return ans_len, f1_avg_len


def search_type_question(dataset,type): 
  where = when = what = who = which = how = whom = why = other = 0 
  for i in range(0, len(dataset[type])): 
      if dataset[type][i]["question"].lower().find("where")!=-1: 
        where +=1 
      elif dataset[type][i]["question"].lower().find("when")!=-1: 
        when += 1 
      elif dataset[type][i]["question"].lower().find("what")!=-1: 
        what +=1 
      elif dataset[type][i]["question"].lower().find("whom")!=-1: 
        who +=1 
      elif dataset[type][i]["question"].lower().find("which")!=-1: 
        which +=1 
      elif dataset[type][i]["question"].lower().find("how")!=-1: 
        how+=1 
      elif dataset[type][i]["question"].lower().find("who")!=-1: 
        whom+=1 
      elif dataset[type][i]["question"].lower().find("why")!=-1: 
        why+=1 
      else: 
        other+=1 
  question_list=[where,when,what,who,which,how,whom,other] 
  return question_list


def computes_f1_q(dataset, f1_raw):
  # returns a list of dictionaries with length of question and f1 score
  f1_len_q = []

  for article in dataset['data']:
      for p in article['paragraphs']:
        for qa in p['qas']:
          qid = qa['id']
          # get length of questions (in words)
          q_len = len(qa['question'].split())
          f1_score = f1_raw[qid]
          f1_len_q.append({'len': q_len, 'f1': f1_score})

  return f1_len_q


def computes_f1_context(dataset, f1_raw):
  # returns a list of dictionaries with length (in words) of context and f1 score
  f1_len = []

  for article in dataset['data']:
      for p in article['paragraphs']:
        c = p['context']
        for qa in p['qas']:
          qid = qa['id']
          f1_score = f1_raw[qid]
          # get lengths for answers
          cont_len = len(c.split())
          f1_len.append({'len': cont_len, 'f1': f1_score})

  return f1_len


def get_data_from_len(dataset, preds, qlen):
  fix_len_ans = []
  for article in dataset['data']:
    for p in article['paragraphs']:
      cont = p['context']
      for qa in p['qas']:
        q = qa['question']
        qid = qa['id']
        for a in qa['answers']:
          if normalize_answer(a['text']):
            ans = a['text']
            s = a['answer_start']
            if qid not in preds:
              print('Missing prediction for %s' % qid)
              continue
            a_pred = preds[qid]
            if len(ans) == qlen: 
              fix_len_ans.append({'context': cont, 
                                  'question': q, 
                                  'answer': ans,
                                  'answer_start': s,
                                  'answer_pred': a_pred})
  return fix_len_ans