import torch, math, os
from transformers import BertTokenizer, BertForNextSentencePrediction
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('built tokenizer')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.eval()
print('built model')

def get_logits(firstsentence, secondsentence):
    global tokenizer, model

    firstwords = word_tokenize(firstsentence)
    secondwords = word_tokenize(secondsentence)

    # we're limited to 512 words, and BERT tokenization is more
    # verbose than NLTK, so I err on the side of caution

    if len(firstwords) + len(secondwords) > 250:
        firstcap = min(len(firstwords), 120)
        firstsentence = ' '.join(firstwords[0: firstcap])
        secondcap = min(len(secondwords), 120)
        secondsentence = ' '.join(secondwords[0 : secondcap])

    encoding = tokenizer.encode_plus(firstsentence, secondsentence, return_tensors = 'pt', max_seq_length = 255)
    loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))

    return loss, logits

def sentence_concat(a, b):
    if len(a) < 1:
        return b
    else:
        return a + ' ' + b

def weak_softmax(logits):
    '''
    I'm writing this rather than using BERT-calculated loss
    because I want a gentler transition  between low probs and
    high probs, which I can produce by using exponentiation
    of 1.2 instead of e (natural log) in the softmax.

    :param logits: a tensor produced by BERT
    :return: probability of the first category after softmax
    '''

    poslogit = logits[0, 0]
    neglogit = logits[0, 1]

    pospart = math.pow(1.2, poslogit)
    negpart = math.pow(1.2, neglogit)

    posprob = pospart / (pospart + negpart)

    return posprob

def record_transitions(path2file, use_sent_boundaries):

    with open(path2file, encoding = 'utf-8') as f:
        filestring = f.read()
    print('read file')
    filestring = filestring.replace('\n', ' ')
    filestring = filestring.replace('\r', ' ')
    filestring = filestring.replace('\t', ' ')
    filestring = filestring.replace('—', ' ')
    filestring = filestring.replace('_', ' ')

    if use_sent_boundaries:
        sentences = sent_tokenize(filestring)
        usable_sentences = []
        thissent = ""

        for s in sentences:
            sentwords = word_tokenize(s)
            sentlen = len(sentwords)
            alreadylen = len(word_tokenize(thissent))
            netlen = sentlen + alreadylen

            if len(sentwords) > 1:
                last_token = sentwords[-1]
            else:
                last_token = ""

            if last_token == "''" or last_token == "'":
                endswithquote = True
            else:
                endswithquote = False

            if netlen <= 5:
                thissent = sentence_concat(thissent, s)
                continue
            elif not endswithquote and netlen <= 16:
                thissent = sentence_concat(thissent, s)
                continue
            elif netlen < 115:
                usable_sentences.append(sentence_concat(thissent, s))
                thissent = ""
            else:
                todivide = sentence_concat(thissent, s)
                wordstodivide = word_tokenize(todivide)
                midpoint = int(len(wordstodivide) / 2)
                firstpart = wordstodivide[0: midpoint]
                secondpart = wordstodivide[midpoint: ]
                if len(firstpart) > 115:
                    firstpart = firstpart[0 : 115]
                if len(secondpart) > 115:
                    secondpart = secondpart[0: 115]

                usable_sentences.append(' '.join(firstpart))
                usable_sentences.append(' '.join(secondpart))
                thissent = ""

        if len(thissent) > 2:
            usable_sentences.append(thissent)

    else:
        usable_sentences = []
        filewords = word_tokenize(filestring)

        for idx in range(0, len(filewords), 180):
            cap = idx + 180
            if cap > len(filewords):
                cap = len(filewords)
            usable_sentences.append(' '.join(filewords[idx : cap]))

    print('made usable sentences: length ' + str(len(usable_sentences)))
    print('now processing')

    probabilities = []
    lengths = []
    backprobs = []  # one tenth of the sentences are compared to a sent 100 back
    pairedprobs = []  # the actual probabilities for those tenth
    logits0 = []
    logits1 = []

    for idx, nextsent in enumerate(usable_sentences):
        nextsentwords = word_tokenize(nextsent)
        leninwords = len(nextsentwords)

        if idx == 0:
            probabilities.append(0)
            lengths.append(leninwords)
            backprobs.append(float('nan'))
            pairedprobs.append(float('nan'))
            logits0.append(0)
            logits1.append(0)
            continue
        elif idx > 100 and idx % 10 == 2:
            backsent = usable_sentences[idx - 100]
            loss, logits = get_logits(backsent, nextsent)
            myloss = weak_softmax(logits)  # calculated with a weaker softmax
            backprobs.append(myloss)
        else:
            backprobs.append(float('nan'))

        if idx % 100 == 1:
            print(idx)

        prevsent = usable_sentences[idx - 1]
        loss, logits = get_logits(prevsent, nextsent)
        myloss = weak_softmax(logits) # calculated with a weaker softmax

        if idx > 100 and idx % 10 == 2:
            pairedprobs.append(myloss)
        else:
            pairedprobs.append(float('nan'))

        probabilities.append(myloss)
        lengths.append(leninwords)
        logits0.append(float(logits[0,0]))
        logits1.append(float(logits[0,1]))

    if sum(~np.isnan(backprobs)) > 1:
        print(np.nanmean(pairedprobs) - np.nanmean(backprobs))

    return lengths, probabilities, usable_sentences, backprobs, logits0, logits1

meta = pd.read_csv('../metadata.tsv', sep = '\t')
meta = meta.sample(frac = 1)

testyet = False

for fileid in meta.guten_id:
    if not testyet:
        fileid = 'testing'
        testyet = True
    print()
    print(fileid)

    inpath = '../wholeclean/' + fileid + '.txt'
    outpath = 'sentprobs/' + fileid + '.tsv'

    if os.path.exists(outpath):
        continue
    else:
        lengths, probabilities, usable_sentences, backprobs, logits0, logits1 = record_transitions(inpath, True) # True == use sent bounds

        with open(outpath, mode='w', encoding='utf-8') as outfile:
            outfile.write('lengthinwords\tbertprob\tsentence\tlogit0\tlogit1\tback100\n')
            for a, b, c, d, e, f in zip(lengths, probabilities, usable_sentences, backprobs, logits0, logits1):
                c = c.replace('"', "''")
                outfile.write(str(a) + '\t' + str(b) + '\t' + c + '\t' + str(e) + '\t' + str(f) + '\t' + str(d) + '\n')