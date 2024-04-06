import csv
import json
import itertools
import random
from typing import Union, Callable

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def load_datasets(data_directory: str) -> "Union[dict, dict]":
    """
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize_w2v(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[: max_words - 1]

    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


def collate_cbow(batch):
    """
    """
    sources = []
    targets = []

    for s, t in batch:
        sources.append(s)
        targets.append(t)

    sources = torch.tensor(sources, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)

    return sources, targets


def train_w2v(model, optimizer, loader, device):
    """
    Code to train the model. See usage at the end.
    """
    model.train()

    for x, y in tqdm(loader, miniters=20, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)

        loss = F.cross_entropy(y_pred, y)
        loss.backward()

        optimizer.step()

    return loss


class Word2VecDataset(torch.utils.data.Dataset):
    """
    Dataset is needed in order to use the DataLoader. See usage at the end.
    """

    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets
        assert len(self.sources) == len(self.targets)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx]

def load_glove_embeddings(file_path: str) -> "dict[str, np.ndarray]":
    """
    Loads trained GloVe embeddings downloaded from:
        https://nlp.stanford.edu/projects/glove/
    """
    word_to_embedding = {}
    with open(file_path, "r") as f:
        for line in f:
            word, raw_embeddings = line.split()[0], line.split()[1:]
            embedding = np.array(raw_embeddings, dtype=np.float64)
            word_to_embedding[word] = embedding
    return word_to_embedding


def load_professions(file_path: str) -> "list[str]":
    """
    Loads profession words from the BEC-Pro dataset. For more information on BEC-Pro,
    see:
        https://arxiv.org/abs/2010.14534
    """
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip the header.
        professions = [row[1] for row in reader]
    return professions


def load_gender_attribute_words(file_path: str) -> "list[list[str]]":
    """
    Loads the gender attribute words from: https://aclanthology.org/N18-2003/
    """
    with open(file_path, "r") as f:
        gender_attribute_words = json.load(f)
    return gender_attribute_words


def compute_partitions(XY: "list[str]") -> "list[tuple]":
    """
    Computes all of the possible partitions of X union Y into equal sized sets.

    Parameters
    ----------
    XY: list of strings
        The list of all target words.

    Returns
    -------
    list of tuples of strings
        List containing all of the possible partitions of X union Y into equal sized
        sets.
    """
    return list(itertools.combinations(XY, len(XY) // 2))


def p_value_permutation_test(
    X: "list[str]",
    Y: "list[str]",
    A: "list[str]",
    B: "list[str]",
    word_to_embedding: "dict[str, np.array]",
) -> float:
    """
    Computes the p-value for a permutation test on the WEAT test statistic.

    Parameters
    ----------
    X: list of strings
        List of target words.
    Y: list of strings
        List of target words.
    A: list of strings
        List of attribute words.
    B: list of strings
        List of attribute words.
    word_to_embedding: dict of {str: np.array}
        Dict containing the loaded GloVe embeddings. The dict maps from words
        (e.g., 'the') to corresponding embeddings.

    Returns
    -------
    float
        The computed p-value for the permutation test.
    """
    # Compute the actual test statistic.
    s = weat_differential_association(X, Y, A, B, word_to_embedding, weat_association)

    XY = X + Y
    partitions = compute_partitions(XY)

    total = 0
    total_true = 0
    for X_i in partitions:
        # Compute the complement set.
        Y_i = [w for w in XY if w not in X_i]

        s_i = weat_differential_association(
            X_i, Y_i, A, B, word_to_embedding, weat_association
        )

        if s_i > s:
            total_true += 1
        total += 1

    p = total_true / total

    return p



def build_current_surrounding_pairs(indices: "list[int]", window_size: int = 2):
    array_len = len(indices)
    surroundings = []
    currents = []
    for i in range(window_size, array_len - window_size):
        currents.append(indices[i])
        surround = []
        for j in range(0 - window_size, 0 + window_size + 1):
            if (j == 0):
                continue
            surround.append(indices[i + j])
        surroundings.append(surround)
    return surroundings, currents


def expand_surrounding_words(ix_surroundings: "list[list[int]]", ix_current: "list[int]"):
    # TODO: your work here
    surr = []
    curr = []
    for i in range(len(ix_surroundings)):
        for j in range(len(ix_surroundings[i])):
            surr.append(ix_surroundings[i][j])
            curr.append(ix_current[i])
    return surr, curr


def cbow_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    # TODO: your work here
    source = []
    target = []
    for ind_list in indices_list:
        surrounding, current = build_current_surrounding_pairs(ind_list, window_size=window_size)
        source += surrounding
        target += current
    return source, target


def skipgram_preprocessing(indices_list: "list[list[int]]", window_size: int = 2):
    source = []
    target = []
    for ind_list in indices_list:
        surrounding, current = build_current_surrounding_pairs(ind_list, window_size=window_size)
        surr, curr = expand_surrounding_words(surrounding, current)
        source += surr
        target += curr
    return source, target


class SharedNNLM:
    def __init__(self, num_words: int, embed_dim: int):
        """
        SkipGram and CBOW actually use the same underlying architecture,
        which is a simplification of the NNLM model (no hidden layer)
        and the input and output layers share the same weights. You will
        need to implement this here.

        Notes
        -----
          - This is not a nn.Module, it's an intermediate class used
            solely in the SkipGram and CBOW modules later.
          - Projection does not have a bias in word2vec
        """

        # TODO vvvvvv
        self.embedding = nn.Embedding(num_words, embed_dim)
        self.projection = nn.Linear(embed_dim, num_words, bias=False)

        # TODO ^^^^^
        self.bind_weights()

    def bind_weights(self):
        """
        Bind the weights of the embedding layer with the projection layer.
        This mean they are the same object (and are updated together when
        you do the backward pass).
        """
        emb = self.get_emb()
        proj = self.get_proj()

        proj.weight = emb.weight

    def get_emb(self):
        return self.embedding

    def get_proj(self):
        return self.projection


class SkipGram(nn.Module):
    """
    Use SharedNNLM to implement skip-gram. Only the forward() method differs from CBOW.
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        emb = self.emb(x)
        output = self.proj(emb)
        return output


class CBOW(nn.Module):
    """
    Use SharedNNLM to implement CBOW. Only the forward() method differs from SkipGram,
    as you have to sum up the embedding of all the surrounding words (see paper for details).
    """

    def __init__(self, num_words: int, embed_dim: int = 100):
        """
        Parameters
        ----------
        num_words: int
            The number of words in the vocabulary.
        embed_dim: int
            The dimension of the word embeddings.
        """
        super().__init__()

        self.nnlm = SharedNNLM(num_words, embed_dim)
        self.emb = self.nnlm.get_emb()
        self.proj = self.nnlm.get_proj()

    def forward(self, x: torch.Tensor):
        # TODO: your work here
        emb = self.emb(x)
        sum_emb = torch.sum(emb, dim=1)
        output = self.proj(sum_emb)
        return output


def compute_topk_similar(
    word_emb: torch.Tensor, w2v_emb_weight: torch.Tensor, k
) -> list:
    # TODO: your work here
    word_emb = word_emb / torch.norm(word_emb)
    w2v_emb_weight = w2v_emb_weight / torch.norm(w2v_emb_weight, dim=1, keepdim=True)
    cos_sim = torch.mm(word_emb, w2v_emb_weight.t())
    _, topk_indice = torch.topk(cos_sim, k+1, dim=1)
    topk_indice = topk_indice[0][1:].tolist()
    
    return topk_indice


@torch.no_grad()
def retrieve_similar_words(
    model: nn.Module,
    word: str,
    index_map: "dict[str, int]",
    index_to_word: "dict[int, str]",
    k: int = 5,
) -> "list[str]":
    model.eval()
    word_index = index_map[word]
    wordemb = model.emb(torch.tensor([word_index]))
    topk_indice = compute_topk_similar(wordemb, model.emb.weight, k)
    similar_words = [index_to_word[ind] for ind in topk_indice]

    return similar_words


@torch.no_grad()
def word_analogy(
    model: nn.Module,
    word_a: str,
    word_b: str,
    word_c: str,
    index_map: "dict[str, int]",
    index_to_word: "dict[int, str]",
    k: int = 5,
) -> "list[str]":
    # TODO: your work here
    a_ind = index_map[word_a]
    b_ind = index_map[word_b]
    c_ind = index_map[word_c]
    model.eval()
    a_emb = model.emb(torch.tensor([a_ind]))
    b_emb = model.emb(torch.tensor([b_ind]))
    c_emb = model.emb(torch.tensor([c_ind]))
    ana_emb = a_emb - b_emb + c_emb
    topk_indice = compute_topk_similar(ana_emb, model.emb.weight, k)
    similar_words = [index_to_word[ind] for ind in topk_indice]
    return similar_words

def compute_gender_subspace(
    word_to_embedding: "dict[str, np.array]",
    gender_attribute_words: "list[tuple[str, str]]",
    n_components: int = 1,
) -> np.array:
    # TODO: your work here
    mean_embedding = [] 
    for words in gender_attribute_words:
        a_emb = word_to_embedding[words[0]]
        b_emb = word_to_embedding[words[1]]
        mean = (a_emb + b_emb) / 2
        cos_a = a_emb - mean
        cos_b = b_emb - mean
        mean_embedding.append(cos_a)
        mean_embedding.append(cos_b)
    mean_embedding =  np.array(mean_embedding)
    pca = PCA(n_components=n_components)
    pca.fit(mean_embedding)
    gender_subspace = pca.components_
    return gender_subspace


def project(a: np.array, b: np.array) -> "tuple[float, np.array]":
    scalar = np.dot(a, b)/ np.dot(b, b)
    vector_projection = scalar * b
    return scalar, vector_projection


def compute_profession_embeddings(
    word_to_embedding: "dict[str, np.array]", professions: "list[str]"
) -> "dict[str, np.array]":
    result = {}
    for profession in professions:
        splitt = profession.split()
        embedding = [word_to_embedding[word] for word in splitt]
        embeddings = np.mean(embedding, axis = 0)
        result[profession] = embeddings
    return result


def compute_extreme_words(
    words: "list[str]",
    word_to_embedding: "dict[str, np.array]",
    gender_subspace: np.array,
    k: int = 10,
    max_: bool = True,
) -> "list[str]":
    sort_later = {}
    for word in words:
        scalar_coef, _ = project(word_to_embedding[word], gender_subspace[0])
        sort_later[word] = scalar_coef
    if max_ :
        sorted_words = sorted(sort_later, key=sort_later.get, reverse=True)
    else:
        sorted_words = sorted(sort_later, key=sort_later.get, reverse=False)
    result = sorted_words[:k]
    return result


def cosine_similarity(a: np.array, b: np.array) -> float:
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def compute_direct_bias(
    words: "list[str]",
    word_to_embedding: "dict[str, np.array]",
    gender_subspace: np.array,
    c: float = 0.25,
):
    # TODO: your work here
    result = 0
    N = len(words)
    gender_space = gender_subspace[0]
    for word in words:
        w = word_to_embedding[word]
        result += abs(cosine_similarity(w, gender_space))**c
    result /= N
    return result


def weat_association(
    w: str, A: "list[str]", B: "list[str]", word_to_embedding: "dict[str, np.array]"
) -> float:
    # TODO: your work here
    # TODO: your work here
    A_Sim = [cosine_similarity(word_to_embedding[w], word_to_embedding[word]) for word in A]
    B_Sim = [cosine_similarity(word_to_embedding[w], word_to_embedding[word]) for word in B]
    return (sum(A_Sim)/len(A_Sim)) - (sum(B_Sim)/len(B_Sim))



def weat_differential_association(
    X: "list[str]",
    Y: "list[str]",
    A: "list[str]",
    B: "list[str]",
    word_to_embedding: "dict[str, np.array]",
    weat_association_func: Callable,
) -> float:
    # TODO: your work here
    return sum(weat_association_func(w, A, B, word_to_embedding) for w in X) - sum(weat_association_func(w, A, B, word_to_embedding) for w in Y)


def debias_word_embedding(
    word: str, word_to_embedding: "dict[str, np.array]", gender_subspace: np.array
) -> np.array:
    result = word_to_embedding[word] - np.dot(word_to_embedding[word], gender_subspace[0])*gender_subspace[0]  
    return result


def hard_debias(
    word_to_embedding: "dict[str, np.array]",
    gender_attribute_words: "list[str]",
    n_components: int = 1,
) -> "dict[str, np.array]":
    # TODO: your work here
    gender_subspace = compute_gender_subspace( word_to_embedding=word_to_embedding,  gender_attribute_words=gender_attribute_words, n_components=n_components)
    result = {}
    for w in word_to_embedding:
        result[w] = debias_word_embedding(w, word_to_embedding, gender_subspace)
    return result


if __name__ == "__main__":
    random.seed(2022)
    torch.manual_seed(2022)

    # Parameters (you can change them)
    sample_size = 2500  # Change this if you want to take a subset of data for testing
    batch_size = 64
    n_epochs = 2
    num_words = 50000

    # Load the data
    data_path = "../input/a1-data"  # Use this for kaggle
    # data_path = "data"  # Use this if running locally

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################### PART 1: TEST CODE ######################
    print("=" * 80)
    print("Running test code for part 1")
    print("-" * 80)

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets(data_path)
    if sample_size is not None:
        for key in ["premise", "hypothesis", "label"]:
            train_raw[key] = train_raw[key][:sample_size]
            valid_raw[key] = valid_raw[key][:sample_size]

    full_text = (
        train_raw["premise"]
        + train_raw["hypothesis"]
        + valid_raw["premise"]
        + valid_raw["hypothesis"]
    )

    # Process into indices
    tokens = tokenize_w2v(full_text)

    word_counts = build_word_counts(tokens)
    word_to_index = build_index_map(word_counts, max_words=num_words)
    index_to_word = {v: k for k, v in word_to_index.items()}

    text_indices = tokens_to_ix(tokens, word_to_index)

    # Train CBOW
    sources_cb, targets_cb = cbow_preprocessing(text_indices, window_size=2)
    loader_cb = DataLoader(
        Word2VecDataset(sources_cb, targets_cb),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cbow,
    )

    model_cb = CBOW(num_words=len(word_to_index), embed_dim=200).to(device)
    optimizer = torch.optim.Adam(model_cb.parameters())

    for epoch in range(n_epochs):
        loss = train_w2v(model_cb, optimizer, loader_cb, device=device).item()
        print(f"Loss at epoch #{epoch}: {loss:.4f}")

    # Training Skip-Gram

    # TODO: your work here
    model_sg = "TODO: use SkipGram"

    # RETRIEVE SIMILAR WORDS
    word = "man"

    similar_words_cb = retrieve_similar_words(
        model=model_cb,
        word=word,
        index_map=word_to_index,
        index_to_word=index_to_word,
        k=5,
    )

    similar_words_sg = retrieve_similar_words(
        model=model_sg,
        word=word,
        index_map=word_to_index,
        index_to_word=index_to_word,
        k=5,
    )

    print(f"(CBOW) Words similar to '{word}' are: {similar_words_cb}")
    print(f"(Skip-gram) Words similar to '{word}' are: {similar_words_sg}")

    # COMPUTE WORDS ANALOGIES
    a = "man"
    b = "woman"
    c = "girl"

    analogies_cb = word_analogy(
        model=model_cb,
        word_a=a,
        word_b=b,
        word_c=c,
        index_map=word_to_index,
        index_to_word=index_to_word,
    )
    analogies_sg = word_analogy(
        model=model_sg,
        word_a=a,
        word_b=b,
        word_c=c,
        index_map=word_to_index,
        index_to_word=index_to_word,
    )

    print(f"CBOW's analogies for {a} - {b} + {c} are: {analogies_cb}")
    print(f"Skip-gram's analogies for {a} - {b} + {c} are: {analogies_sg}")

    # ###################### PART 1: TEST CODE ######################

    # Prefilled code showing you how to use the helper functions
    word_to_embedding = load_glove_embeddings("data/glove/glove.6B.300d.txt")

    professions = load_professions("data/professions.tsv")

    gender_attribute_words = load_gender_attribute_words(
        "data/gender_attribute_words.json"
    )

    # === Section 2.1 ===
    gender_subspace = "your work here"

    # === Section 2.2 ===
    a = "your work here"
    b = "your work here"
    scalar_projection, vector_projection = "your work here"

    # === Section 2.3 ===
    profession_to_embedding = "your work here"

    # === Section 2.4 ===
    positive_profession_words = "your work here"
    negative_profession_words = "your work here"

    print(f"Max profession words: {positive_profession_words}")
    print(f"Min profession words: {negative_profession_words}")

    # === Section 2.5 ===
    direct_bias_professions = "your work here"

    # === Section 2.6 ===

    # Prepare attribute word sets for testing
    A = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
    B = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

    # Prepare target word sets for testing
    X = ["doctor", "mechanic", "engineer"]
    Y = ["nurse", "artist", "teacher"]

    word = "doctor"
    weat_association = "your work here"
    weat_differential_association = "your work here"

    # === Section 3.1 ===
    debiased_word_to_embedding = "your work here"
    debiased_profession_to_embedding = "your work here"

    # === Section 3.2 ===
    direct_bias_professions_debiased = "your work here"

    print(f"DirectBias Professions (debiased): {direct_bias_professions_debiased:.2f}")

    X = [
        "math",
        "algebra",
        "geometry",
        "calculus",
        "equations",
        "computation",
        "numbers",
        "addition",
    ]

    Y = [
        "poetry",
        "art",
        "dance",
        "literature",
        "novel",
        "symphony",
        "drama",
        "sculpture",
    ]

    # Also run this test for debiased profession representations.
    p_value = "your work here"

    print(f"p-value: {p_value:.2f}")
