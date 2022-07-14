from itertools import product, chain
from collections import Counter
import nltk
import pandas as pd
import networkx as nx
import community
import generalized

stops = set(nltk.corpus.stopwords.words('english')) \
        | {'social', 'network', 'analysis', 'u', 'using', 'use', 'vs.'}
wnl = nltk.WordNetLemmatizer()

df = pd.read_csv('affiliation.csv')
df = df.rename(columns={'Unnamed: 0': 'paper'})

df['tokens'] = df['paper'].apply(nltk.word_tokenize).apply(nltk.pos_tag)

df['tokens'] = df['tokens'].apply(lambda t:
        [wnl.lemmatize(w.lower()) for w,pos in t
         if pos.startswith('N') or pos.startswith('J')])

df['tokens'] = df['tokens'].apply(lambda x: set(x) - stops)

terms = Counter(chain.from_iterable(df['tokens']))
counts = pd.Series(terms).sort_values()
top_counts = set(counts[counts > 1].index)

df['tokens'] = df['tokens'].apply(lambda x: x & top_counts)
df_ok = df[df.tokens.str.len() > 0]
df_ok.set_index('paper', inplace=True)

edges = list(chain.from_iterable(product(df_ok.iloc[i,:-1][df_ok.iloc[i,:-1]==1].index, df_ok.iloc[i,-1]) for i in range(df_ok.shape[0])))
G = nx.Graph(edges)
G = nx.subgraph(G, sorted(nx.connected_components(G), key=len)[-1])
A = generalized.generalized_similarity(G)
B = A[0] if 'life' in A[1] else A[1]

N = nx.Graph([(n1, n2, {'weight': w['weight']}) for n1, n2, w
              in B.edges(data=True) if w['weight'] >= 0.75 and n1 != n2])
parts_n = community.best_partition(N)
nx.set_node_attributes(N, parts_n, 'part')
nx.set_node_attributes(N, df.sum(), "s")
nx.write_graphml(N, "names.graphml")

# T = nx.Graph([(n1, n2, {'weight': w['weight']}) for n1, n2, w
#               in A[1].edges(data=True) if w['weight'] >= 0.75 and n1 != n2])
# parts_t = community.best_partition(T)
# nx.set_node_attributes(T, parts_t, 'part')
# nx.set_node_attributes(T, counts, "s")
# nx.write_graphml(T, "terms.graphml")
