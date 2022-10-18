def gen_model(users):

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd

    #gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    from gensim.models import TfidfModel

    #spacy
    import spacy

    df = pd.read_csv('../oportunidades.csv')
    data = list(df['opo_texto'])

    def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return [" ".join([token.lemma_ for token in nlp(text) if token.pos_ in allowed_postags]) for text in texts]

    lemmatized_texts = lemmatization(data)

    def gen_words(texts):
        return [gensim.utils.simple_preprocess(text, deacc=True) for text in texts]

    data_words = gen_words(lemmatized_texts)

    #BIGRAMS
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=50)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)

    def make_bigrams(texts):
        return [bigram[doc] for doc in texts]

    data_bigrams = make_bigrams(data_words)

    #DF FILTER
    id2word = corpora.Dictionary(data_bigrams)
    texts = data_bigrams
    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value=0.4
    words = []
    words_missing_in_tfidf = []

    for i in range(0, len(corpus)):
        bow = corpus[i]
        tfidf_ids = [id for id,value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        hfreq_words = [id for id in bow_ids if tfidf.dfs[id]/len(corpus) > low_value]
        drops = hfreq_words + words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

        new_bow = [b for b in bow if b[0] not in hfreq_words and b[0] not in words_missing_in_tfidf]       
        corpus[i] = new_bow

    #TF-IDF FILTER
    id2word = corpora.Dictionary(data_bigrams)
    texts = data_bigrams
    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value=0.02
    words = []
    words_missing_in_tfidf = []

    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = [] #reinitialize to be safe. You can skip this.
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]       
        corpus[i] = new_bow

    lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=users,
                                            random_state=100,
                                            chunksize=100,
                                            passes=10)

    lda_model.save(f'models/lda_model{users}.model')