from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint

# Sample documents
documents = [
    'HEGEL/hegel.txt'
]

# Tokenize the documents into words
tokenized_documents = [doc.split() for doc in documents]

# Create a dictionary mapping words to unique ids
dictionary = corpora.Dictionary(tokenized_documents)

# Convert the tokenized documents to a bag-of-words representation
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Build the LDA model
lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

# Print the topics and their associated words
print(lda_model.print_topics())

# Optionally, you can assign topics to documents
# for i, doc in enumerate(corpus):
    # print(f"Document {i + 1} - Topic Distribution: {lda_model[doc]}")
