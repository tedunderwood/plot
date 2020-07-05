Exploratory research on change across narrative time
====================================================

This is not a fully developed project yet, just some scattered elements of code and data I used to produce an exploratory blog post. There's no paper to reproduce, so it's not "reproducible" yet.

BERT stuff
----------

The code for using BERT is in sentences/sentence_probabilities.py, but most of it is fussy stuff about dividing sentences and ensuring they don't overrun the window.

The actual core of the code for BERT is shockingly simple:

    import torch
    from transformers import BertTokenizer, BertForNextSentencePrediction

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('built tokenizer')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    model.eval()
    print('built model')

    encoding = tokenizer.encode_plus(firstsentence, secondsentence, return_tensors = 'pt', max_seq_length = 255)
    loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))

The [documentation for the HuggingFace implementation of **transformers**](https://huggingface.co/transformers/model_doc/bert.html#bertfornextsentenceprediction) is actually fairly comprehensible.
