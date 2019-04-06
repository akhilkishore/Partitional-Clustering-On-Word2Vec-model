#import word2vec model to deal with models
#actually i used fastText model introduced by facebook for phrase handling instead of words.
#it works over wor2vec foundation

from gensim.models import Word2Vec 

#import cluster from sklearn

from sklearn import cluster
import matplotlib.pyplot as plt

#calling model

model_F = Word2Vec.load("FastTextCount1.model")
vocab_F = model_F[model_F.wv.vocab]
X = model_F[model_F.wv.vocab]

#calling kmeans from sklearn

kmeans = cluster.KMeans(n_clusters=50)
kmeans.fit(vocab_F)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

l = list(model_F.wv.vocab)

print(len(l))
print(len(vocab_F))
print (len(kmeans.labels_))

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow') 
plt.show()
