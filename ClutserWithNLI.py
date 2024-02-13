import torch.cuda
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
import pickle
from  PurposeReader import PurposeReader
import torch


class SaveSentenceEmbeddings:

    def __init__(self,all_sentences, batch_size):
        self.sbert_model = SentenceTransformer('stsb-roberta-base')
        self.all_sentences = all_sentences
        self.batch_size = batch_size

    def add_batch_encodings(self,d, batch):
        embs = self.sbert_model.encode(self.all_sentences)
        for i in range(len(batch)):
            d[batch[i]] =  embs[i]


    def save_embeddings_dict(self,path):
        iterations = len(self.all_sentences) // self.batch_size
        cur_idx = 0
        d = {}
        for i in range(iterations):
            print(f"started batch {i}")
            batch = self.all_sentences[cur_idx : cur_idx + self.batch_size]
            cur_idx += self.batch_size
            self.add_batch_encodings(d, batch)
        #adding the last batch
        self.add_batch_encodings(d, self.all_sentences[cur_idx:])
        with open(path,"wb") as f:
            pickle.dump(d,f)

def load_sentence_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)




if __name__ == "__main__":
    dir = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\"
    PR = PurposeReader()
    purpose_dict = PR.create_purpose_dict(dir)
    sentences = list(purpose_dict.values())[:50000]
    path = "50k_embeddings.pkl"
    #sentences = ["hi hello", "what's up"]
    SSE = SaveSentenceEmbeddings(sentences, 32)
    SSE.save_embeddings_dict(path)
    d = load_sentence_embeddings(path)
    x = 1