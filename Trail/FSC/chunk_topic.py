from gensim import corpora, models
from gensim.utils import simple_preprocess

def chunk_topic(text, num_topics=3):
    docs = [p for p in text.split("\n") if len(p.strip()) > 20]
    tokenized = [simple_preprocess(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]

    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=2)
    topic_assignments = [max(lda.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]

    topic_chunks = {}
    for doc, topic in zip(docs, topic_assignments):
        topic_chunks.setdefault(topic, []).append(doc)

    return [" ".join(group) for group in topic_chunks.values()]
