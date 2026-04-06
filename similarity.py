import numpy as np

# Cosine Similarity
class cosineSimilarity:
    def __init__ (self, embeddings):
        self.embeddings = embeddings
    
    def search(self, vec, top_k):
        vec = vec.reshape(-1)
        numerators = self.embeddings @ vec # Matrix Multiplication

        denom_chunks = np.linalg.norm(self.embeddings, axis = 1)
        denom_vec = np.linalg.norm(vec)

        similarity_scores = numerators / (denom_chunks * denom_vec)

        # Sort from highest to lowest, and get the top_k indecies 
        top_indicies = np.argsort(similarity_scores)[::-1][:top_k] 
        return similarity_scores[top_indicies], top_indicies
    
# Manhattan Similarity
class manhattanSimilarity:
    def __init__ (self, embeddings):
        self.embeddings = embeddings
    
    def search(self, vec, top_k):
        diff = self.embeddings - vec
        similarity_scores = np.sum(np.abs(diff), axis=1)

        # Want lowest to highest and get the top_k indecies 
        top_indicies = np.argsort(similarity_scores)[:top_k]

        return similarity_scores[top_indicies], top_indicies
    
# Euclidean Similarity
class euclideanSimilarity:
    def __init__ (self, embeddings):
        self.embeddings = embeddings
    
    def search(self, vec, top_k):
        diff = self.embeddings - vec
        similarity_scores = np.linalg.norm(diff, axis=1)

        # Want lowest to highest and get the top_k indecies 
        top_indicies = np.argsort(similarity_scores)[:top_k]

        return similarity_scores[top_indicies], top_indicies