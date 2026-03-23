# Copyright [2026] GitHub Copilot.

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline, TextToEmbeddingModelPipeline

import commons
import dataset_manager as dsm


torch.set_grad_enabled(False)

_text2vec = None
_vec2text = None
_device = None


def _get_device():
    global _device
    if _device is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        _device = torch.device(device_name)
    return _device


def _get_text2vec():
    global _text2vec
    if _text2vec is None:
        _text2vec = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder',
            device=_get_device(),
        )
    return _text2vec


def _get_vec2text():
    global _vec2text
    if _vec2text is None:
        _vec2text = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_encoder',
            device=_get_device(),
        )
    return _vec2text


def encode_texts(texts, dataset):
    if len(texts) == 0:
        return np.zeros((0, commons.text_embedding_dim), dtype=np.float32)

    max_seq_len = commons.dataset_max_seq_len[dataset]
    pipeline = _get_text2vec()
    batches = []
    total = len(texts)

    progress_columns = [
        TextColumn('[progress.description]{task.description}'),
        BarColumn(),
        TextColumn('{task.completed}/{task.total} texts'),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns) as progress:
        task_id = progress.add_task(f'Encoding {dataset}', total=total)
        for start in range(0, total, commons.embedding_batch_size):
            batch = texts[start : start + commons.embedding_batch_size]
            embedding = pipeline.predict(
                list(batch),
                source_lang='eng_Latn',
                max_seq_len=max_seq_len,
            )
            batches.append(embedding.detach().cpu().numpy().astype(np.float32))
            progress.update(task_id, advance=len(batch))

    return np.concatenate(batches, axis=0)


def decode_embeddings(embeddings, dataset):
    if len(embeddings) == 0:
        return []

    max_seq_len = commons.dataset_max_seq_len[dataset]
    pipeline = _get_vec2text()
    decoded_texts = []
    embeddings = np.asarray(embeddings, dtype=np.float32)

    for start in range(0, len(embeddings), commons.embedding_batch_size):
        batch = embeddings[start : start + commons.embedding_batch_size]
        decoded_batch = pipeline.predict(
            torch.from_numpy(batch).to(_get_device()),
            target_lang='eng_Latn',
            max_seq_len=max_seq_len,
        )
        decoded_texts.extend(str(text).strip() for text in decoded_batch)

    return decoded_texts


def roundtrip_embeddings(embeddings, dataset):
    decoded_texts = decode_embeddings(embeddings, dataset)
    return encode_texts(decoded_texts, dataset)


def train_network(ds, prefix, es):
    raise NotImplementedError('Training is no longer used. Features are generated with pretrained SONAR models.')


def obtain_features(ds, model_prefix, features_prefix, labels_prefix, data_prefix, es):
    for fold in range(commons.n_folds):
        training_data, training_labels = dsm.get_training(ds, fold)
        filling_data, filling_labels = dsm.get_filling(ds, fold)
        testing_data, testing_labels = dsm.get_testing(ds, fold)

        settings = [
            (training_data, training_labels, commons.training_suffix, False),
            (filling_data, filling_labels, commons.filling_suffix, False),
            (testing_data, testing_labels, commons.testing_suffix, True),
        ]

        for data, labels, suffix, save_raw_data in settings:
            features = encode_texts(data, ds)
            features_filename = commons.data_filename(features_prefix + suffix, es, fold)
            labels_filename = commons.data_filename(labels_prefix + suffix, es, fold)
            np.save(features_filename, features)
            np.save(labels_filename, labels)
            if save_raw_data:
                data_filename = commons.data_filename(data_prefix + suffix, es, fold)
                np.save(data_filename, np.array(data, dtype=object))
