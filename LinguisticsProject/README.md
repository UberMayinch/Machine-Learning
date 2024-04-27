First tokenization is done using NLTK.
It involves cleaning the text, removing non alphanumeric characters (optionally can involve lemmatization and stemming but these are not relevant here)
We do this for each of the philosophers to end up with a corpus of tokens. 

The books considered: 
Hegel:
1. Science of Logic
2. Phenomenology of Spirit

After this I created word embeddings on the corpus and did Latent Dirichilet Allocation on it
