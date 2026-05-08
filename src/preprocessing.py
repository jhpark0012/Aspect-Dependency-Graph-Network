import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # noqa: E402

import torch
import torch.multiprocessing as mp  # noqa: E402
mp.set_sharing_strategy("file_system")  # noqa: E402
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

import pickle
import collections
import numpy as np
import spacy
from scipy.stats import chi2_contingency
from datasets import DatasetDict

import os
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')


def calc_ch2_test(train_full: DatasetDict, domain: str, save_dir: str):
    if domain == 'Rest-TA':
        aspects = ['Overall', 'Value', 'Service', 'Food', 'Atmosphere']
    elif domain == 'Hotel-TA':
        aspects = ['Overall', 'Value', 'Service', 'Location',
                   'Rooms', 'Cleanliness', 'Sleep Quality']
    else:
        raise ValueError(f"Unknown domain: {domain}")

    full_label = [train_full[i]['final_polarity']
                  for i in range(len(train_full))]

    labels_arr = np.array(full_label)
    n_samples, n_aspects = labels_arr.shape

    full_label_oe = np.zeros((n_samples, n_aspects * 2), dtype=int)
    full_label_oe[:, 0::2] = (labels_arr == 0).astype(int)
    full_label_oe[:, 1::2] = (labels_arr != 0).astype(int)

    cols = []
    for aspect in aspects:
        for pol in ['_neg', '_pos']:
            cols.append(aspect + pol)

    n_cols = len(cols)
    chi2_matrix = np.zeros((n_cols, n_cols))
    pval_matrix = np.zeros((n_cols, n_cols))
    cond_prob_matrix = np.zeros((n_cols, n_cols))

    X = full_label_oe
    O11 = np.dot(X.T, X)
    S = X.sum(axis=0)

    O10 = S[:, None] - O11                 # (A=1, B=0)
    O01 = S[None, :] - O11                 # (A=0, B=1)
    O00 = n_samples - O11 - O10 - O01      # (A=0, B=0)

    for i in range(n_cols):
        for j in range(n_cols):
            contingency = np.array([
                [O00[i, j], O01[i, j]],
                [O10[i, j], O11[i, j]]
            ])

            chi2, p, _, _ = chi2_contingency(contingency)
            chi2_matrix[i, j] = chi2
            pval_matrix[i, j] = p

            marginal = O01[i, j] + O11[i, j]
            cond_prob_matrix[i, j] = (
                O11[i, j] / marginal) if marginal > 0 else np.nan

    with open(save_dir / "raw_dependency_matrix.pkl", "wb") as f:
        pickle.dump(cond_prob_matrix, f)


def _last_valid_char_end(offset_mapping):
    valid_offsets = [(s, e) for s, e in offset_mapping if s != e]
    return valid_offsets[-1][1] if valid_offsets else 0


def compute_dependency_adj(doc, offset_mapping) -> torch.Tensor:
    seq_len = len(offset_mapping)
    sub2word = [-1] * seq_len
    word2sub = collections.defaultdict(list)

    doc_tokens = list(doc)
    num_tokens = len(doc_tokens)
    token_idx = 0

    for sub_idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue

        while token_idx < num_tokens and (doc_tokens[token_idx].idx + len(doc_tokens[token_idx])) <= start:
            token_idx += 1

        if token_idx < num_tokens:
            token = doc_tokens[token_idx]
            token_start = token.idx
            token_end = token.idx + len(token)
            if token_start <= start < token_end:
                sub2word[sub_idx] = token_idx
                word2sub[token_idx].append(sub_idx)

    row, col = [], []
    valid_subwords = [i for i, w in enumerate(sub2word) if w != -1]

    row.extend(valid_subwords)
    col.extend(valid_subwords)

    for token in doc_tokens:
        if token.i == token.head.i:
            continue

        wi, wj = token.i, token.head.i
        if wi not in word2sub or wj not in word2sub:
            continue

        for idx_i in word2sub[wi]:
            for idx_j in word2sub[wj]:
                row.extend([idx_i, idx_j])
                col.extend([idx_j, idx_i])

    if not row:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.tensor([row, col], dtype=torch.long)
        values = torch.ones(len(row), dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=(seq_len, seq_len)).coalesce()


class CustomDataset(Dataset):
    def __init__(self, texts, labels, nlp, tokenizer, max_len: int = 512,
                 spacy_batch_size: int = 256, spacy_n_process: int = 10):
        self.texts = list(texts)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.MAX_LEN = max_len
        self.tokenizer = tokenizer
        self.nlp = nlp

        self.data = self._build_all_data(
            spacy_batch_size=spacy_batch_size,
            spacy_n_process=spacy_n_process,
        )

    def _build_all_data(self, spacy_batch_size: int, spacy_n_process: int):
        encodings = self.tokenizer(
            self.texts,
            truncation=True,
            max_length=self.MAX_LEN,
            padding=False,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        parse_texts = []
        for text, offset_mapping in zip(self.texts, encodings["offset_mapping"]):
            char_end = _last_valid_char_end(offset_mapping)
            parse_texts.append(text[:char_end] if char_end > 0 else "")

        docs = self.nlp.pipe(
            parse_texts,
            batch_size=spacy_batch_size,
            n_process=spacy_n_process,
        )

        built_data = []
        iterator = zip(
            encodings["input_ids"],
            encodings["attention_mask"],
            encodings["offset_mapping"],
            docs,
            self.labels,
        )

        for input_ids, attention_mask, offset_mapping, doc, label in tqdm(
            iterator,
            total=len(self.texts),
            desc="Building parsed dataset",
        ):
            input_ids_t = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)
            adj_matrix = compute_dependency_adj(doc, offset_mapping)

            built_data.append({
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "syntax_graph": adj_matrix,
                "labels": label,
            })

        return built_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _pad_sparse_graph(graph: torch.Tensor, target_len: int) -> torch.Tensor:
    if graph.size(0) == target_len:
        return graph

    return torch.sparse_coo_tensor(
        graph.indices(),
        graph.values(),
        size=(target_len, target_len),
    ).coalesce()


def _pad_sparse_graph(graph: torch.Tensor, target_len: int) -> torch.Tensor:
    if graph.size(0) == target_len:
        return graph

    return torch.sparse_coo_tensor(
        graph.indices(),
        graph.values(),
        size=(target_len, target_len),
    ).coalesce()


def custom_collate(batch):
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    max_len = max(x.size(0) for x in input_ids_list)

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0)

    syntax_graphs = [
        _pad_sparse_graph(item["syntax_graph"], max_len)
        for item in batch
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "syntax_graph": syntax_graphs,
    }


class LoadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def get_prep(BASE_DIR, args, train_clean, val_clean, test_clean) -> None:

    print("Loading Tokenizer and SpaCy NLP model ..")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["ner", "lemmatizer", "textcat",
                 "textcat_multilabel", "attribute_ruler"]
    )

    all_texts = list(train_clean["combined_rev"]) + list(
        val_clean["combined_rev"]) + list(test_clean["combined_rev"])
    all_labels = list(train_clean["final_polarity"]) + list(
        val_clean["final_polarity"]) + list(test_clean["final_polarity"])

    all_custom = CustomDataset(
        all_texts,
        all_labels,
        nlp=nlp,
        tokenizer=tokenizer,
        max_len=512,
        spacy_batch_size=256,
        spacy_n_process=10,
    )

    len_train = len(train_clean)
    len_val = len(val_clean)

    train_data = all_custom.data[:len_train]
    val_data = all_custom.data[len_train: len_train + len_val]
    test_data = all_custom.data[len_train + len_val:]

    save_final_data_dir = BASE_DIR / \
        f"Data/{args.domain}/final_data"
    os.makedirs(save_final_data_dir, exist_ok=True)

    print(f"Saving final data to {save_final_data_dir} .. ")
    torch.save(train_data, os.path.join(save_final_data_dir, "train_data.pt"))
    torch.save(val_data, os.path.join(save_final_data_dir, "val_data.pt"))
    torch.save(test_data, os.path.join(save_final_data_dir, "test_data.pt"))

    save_dependency_matrix_dir = BASE_DIR / \
        f"Data/data_info/{args.domain}/"
    os.makedirs(save_dependency_matrix_dir, exist_ok=True)

    train_full = [train_clean[idx] for idx, i in enumerate(
        train_clean['final_polarity']) if -1 not in i]

    calc_ch2_test(train_full=train_full, domain=args.domain,
                  save_dir=save_dependency_matrix_dir)
