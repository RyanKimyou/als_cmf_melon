#!/usr/bin/env python
"""CMF"""
import logging
import argparse
import json
import os
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
from tqdm.auto import tqdm

from implicit.als import AlternatingLeastSquares as ALS
from implicit.bpr import BayesianPersonalizedRanking as BPR
from implicit.evaluation import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, df):
        logging.info("Creating vocab from %d playlists", len(df))

        songs = set(song for songs in df.songs for song in songs)
        logging.info("Found %d unique songs", len(songs))
        self._id_to_song = dict(enumerate(songs))
        self._song_to_id = dict((v, k) for k, v in self._id_to_song.items())

        base_index = len(songs)

        tags = set(tag for tags in df.tags for tag in tags)
        logging.info("Found %d unique tags", len(tags))
        self._id_to_tag = dict((k + base_index, v) for k, v in enumerate(tags))
        self._tag_to_id = dict((v, k) for k, v in self._id_to_tag.items())

    def song_to_id(self, song):
        return self._song_to_id[song]

    def id_to_song(self, idx):
        return self._id_to_song[idx]

    def tag_to_id(self, tag):
        return self._tag_to_id[tag]

    def id_to_tag(self, idx):
        return self._id_to_tag[idx]

    @property
    def num_songs(self):
        return len(self._id_to_song)

    @property
    def size(self):
        return len(self._id_to_song) + len(self._id_to_tag)

    def is_tag(self, idx):
        return idx in self._id_to_tag


def encode_features(df, vocab):
    return [
        [vocab.song_to_id(song) for song in row.songs]
        + [vocab.tag_to_id(tag) for tag in row.tags]
        for row in df.itertuples(index=False)
    ]


def train_and_predict(train_filepath, test_filepath):
    train_df = pd.read_json(train_filepath)
    test_df = pd.read_json(test_filepath)

    tr_songs = train_df.songs.tolist()
    te_songs = test_df.songs.tolist()
    tr_tags = train_df.tags.tolist()
    te_tags = test_df.tags.tolist()

    vocab = Vocabulary(pd.concat([train_df, test_df], ignore_index=True))

    train_data = encode_features(train_df, vocab)
    test_data = encode_features(test_df, vocab)

    # Shuffle train data
    train_data = shuffle(train_data)

    # list of lists -> CSR
    def lil_to_csr(indices, shape):
        data = []
        row_ind = []
        col_ind = []
        for row_idx, row in enumerate(indices):
            for col_idx in row:
                data.append(1)
                row_ind.append(row_idx)
                col_ind.append(col_idx)
        return csr_matrix((data, (row_ind, col_ind)), shape=shape)

    train_csr = lil_to_csr(train_data, (len(train_data), vocab.size))
    test_csr = lil_to_csr(test_data, (len(test_data), vocab.size))

    r = scipy.sparse.vstack([test_csr, train_csr])
    r = csr_matrix(r)

    factors = 512
    alpha = 500.0
    als_model = ALS(factors=factors, regularization=0.1)
    als_model.fit(r.T * alpha)

    song_model = ALS(factors=factors)
    tag_model = ALS(factors=factors)
    song_model.user_factors = als_model.user_factors
    tag_model.user_factors = als_model.user_factors
    song_model.item_factors = als_model.item_factors[: vocab.num_songs]
    tag_model.item_factors = als_model.item_factors[vocab.num_songs :]

    song_rec_csr = test_csr[:, : vocab.num_songs]
    tag_rec_csr = test_csr[:, vocab.num_songs :]

    song_rec = song_model.recommend_all(song_rec_csr, N=100)
    tag_rec = tag_model.recommend_all(tag_rec_csr, N=10)
    tag_rec += vocab.num_songs

    return [
        {
            "id": test_playlist_id,
            "songs": list(map(vocab.id_to_song, song_rec[test_row_idx])),
            "tags": list(map(vocab.id_to_tag, tag_rec[test_row_idx])),
        }
        for test_row_idx, test_playlist_id in enumerate(tqdm(test_df.id))
    ]


def main():
    parser = argparse.ArgumentParser(description="ALS CMF")

    parser.add_argument(
        "--train_file", type=str, help="training dataset file", required=True
    )

    parser.add_argument(
        "--test_file", type=str, help="test dataset file", required=True
    )

    parser.add_argument("--result_file", type=str, help="output file")

    args = parser.parse_args()

    train_filepath = os.path.expanduser(args.train_file)
    test_filepath = os.path.expanduser(args.test_file)
    result_filepath = os.path.expanduser(args.result_file)

    predictions = train_and_predict(train_filepath, test_filepath)

    Path(result_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(result_filepath, "w") as outfile:
        json.dump(predictions, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
