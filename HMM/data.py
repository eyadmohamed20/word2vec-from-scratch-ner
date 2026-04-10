import re
import pickle
import numpy as np
from datasets import load_dataset

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
NUM_CLASSES = len(NER_LABELS)


def preprocess_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r"[^a-z0-9\-]", "", token)
    return token


def get_context_vector(sentence_vecs, idx, window):
    """
    Concatenate embeddings of [i-window, ..., i, ..., i+window].
    Out-of-bounds positions are zero-padded.
    """
    embed_dim = sentence_vecs.shape[1]
    parts = []
    for offset in range(-window, window + 1):
        j = idx + offset
        if 0 <= j < len(sentence_vecs):
            parts.append(sentence_vecs[j])
        else:
            parts.append(np.zeros(embed_dim, dtype=np.float32))  # padding
    return np.concatenate(parts)  # shape: (embed_dim * (2*window+1),)


def load_data(vocab_path: str, embeddings_path: str, window: int = 2):
    """
    Returns:
        train, val, test  – each is (X, y)
            X : (N, embed_dim * (2*window+1))  – context window vectors
            y : (N,)                            – integer NER tags
        word2idx, embeddings, window
    """
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
    word2idx   = vocab_data["word2idx"]
    embeddings = np.load(embeddings_path)
    embed_dim  = embeddings.shape[1]

    dataset = load_dataset("lhoestq/conll2003", trust_remote_code=False)

    def split_to_arrays(split_name):
        X_list, y_list = [], []
        for example in dataset[split_name]:
            tokens   = example["tokens"]
            ner_tags = example["ner_tags"]

            # Build embedding matrix for this sentence
            sent_vecs = []
            for token in tokens:
                clean = preprocess_token(token)
                if clean in word2idx:
                    sent_vecs.append(embeddings[word2idx[clean]])
                else:
                    sent_vecs.append(np.zeros(embed_dim, dtype=np.float32))
            sent_vecs = np.array(sent_vecs)  # (T, D)

            # Build context window vector for each token
            for i, tag in enumerate(ner_tags):
                X_list.append(get_context_vector(sent_vecs, i, window))
                y_list.append(tag)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    train = split_to_arrays("train")
    val   = split_to_arrays("validation")
    test  = split_to_arrays("test")

    return train, val, test, word2idx, embeddings, window