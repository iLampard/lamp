import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class SentenceEncoder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode_event(self, event_text):
        emb = self.model.encode(event_text)
        print(np.shape(emb))
        return emb



if __name__ == '__main__':
    encoder = SentenceEncoder()
    emb_1 = encoder.encode_event('subject name:CN; object:USA, relation: fight')
    emb_2 = encoder.encode_event('subject name:US Gov; object name: US, relation: make statement')
    print(cos_sim(emb_1, emb_2))