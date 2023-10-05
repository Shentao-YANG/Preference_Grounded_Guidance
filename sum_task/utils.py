import json
import random
import torch
from torch import nn
from datasets import load_dataset
from transformers import CONFIG_MAPPING
import nltk
import numpy as np
from transformers import AutoConfig


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_dict(d): print(json.dumps(d, indent=2), flush=True)


def prints(s, warning=False):
    if warning:
        print(f"[WARNING !!!] {s}", flush=True)
    else:
        print(s, flush=True)


def print_banner(s, symbol="-", front=False, back=False):
    len_s = len(s)
    if front:
        print(symbol * len_s, flush=True)
    print(s, flush=True)
    if back:
        print(symbol * len_s, flush=True)


def _build_one_layer_mlp(in_dim, out_dim, hidden_size):
    W1 = nn.Linear(in_dim, hidden_size)
    A1 = nn.ReLU()
    W2 = nn.Linear(hidden_size, out_dim)
    return nn.Sequential(W1, A1, W2)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.0001)
        m.bias.data.fill_(-0.0001)


summarization_name_mapping = {
    "cnn_dailymail": ("article", "highlights"),
    "xsum": ("document", "summary"),
}


def load_raw_datasets(args):
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    return raw_datasets


def load_pretrained_model_config(args, loader):

    if args.config_name:
        config = loader.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = loader.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        prints("You are instantiating a new config instance from scratch.", warning=True)

    return config


def load_pretrained_tokenizer(args, loader):

    if args.tokenizer_name:
        tokenizer = loader.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = loader.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def load_pretrained_model(args, loader, config, tokenizer, reward_model=False):
    prints(f"Load model {'REWARD' if reward_model else 'POLICY'}: {args.rew_model_name_or_path if reward_model else args.model_name_or_path}", warning=True)

    config = AutoConfig.from_pretrained(args.rew_model_name_or_path) if reward_model else config
    # in case reward model is different from the LM (e.g., T5-small v.s. T5-base)

    if args.model_name_or_path or args.rew_model_name_or_path:
        model = loader.from_pretrained(
            args.rew_model_name_or_path if reward_model else args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path or ".ckpt" in args.rew_model_name_or_path),
            config=config,
        )
    else:
        prints("Training new model from scratch", warning=True)
        model = loader.from_config(config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < args.max_source_length
    ):
        if args.resize_position_embeddings is None:
            prints(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {args.max_source_length}.", warning=True
            )
            model.resize_position_embeddings(args.max_source_length)
        elif args.resize_position_embeddings:
            model.resize_position_embeddings(args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    return model.to(args.device)


def preprocessing_raw_dataset(args, raw_datasets):

    column_names = raw_datasets["train"].column_names

    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}")
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}")

    ########## sub-size the train, val, and test set ##########
    if args.max_train_samples is not None:
        max_train_samples = min(len(raw_datasets["train"]), args.max_train_samples)
        raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))
        prints(f"Downsample train set to {len(raw_datasets['train'])} samples")
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(raw_datasets["validation"]), args.max_eval_samples)
        raw_datasets["validation"] = raw_datasets["validation"].select(range(max_eval_samples))
        prints(f"Downsample validation set to {len(raw_datasets['validation'])} samples")
    if args.max_predict_samples is not None:
        max_predict_samples = min(len(raw_datasets["test"]), args.max_predict_samples)
        raw_datasets["test"] = raw_datasets["test"].select(range(max_predict_samples))
        prints(f"Downsample test set to {len(raw_datasets['test'])} samples")
    ############################################################

    return column_names, text_column, summary_column, raw_datasets


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def get_decoded_preds_labels_from_batch(args, batch, model, tokenizer, greedy_decoding, num_samples, gen_kwargs, for_reward_training):
    if greedy_decoding:
        generated_tokens = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )
    else:
        generated_tokens = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=True,
            num_return_sequences=num_samples,
            **gen_kwargs,
        )
        batch["labels"] = batch["labels"].repeat_interleave(num_samples, dim=0)
        if for_reward_training:
            generated_tokens_mask = (generated_tokens.detach() != tokenizer.pad_token_id).int()

    labels = batch["labels"]

    generated_tokens_np = generated_tokens.cpu().numpy()
    labels = labels.cpu().numpy()

    if args.ignore_pad_token_for_loss:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    if isinstance(generated_tokens_np, tuple):
        generated_tokens_np = generated_tokens_np[0]
    decoded_preds = tokenizer.batch_decode(generated_tokens_np, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    if for_reward_training:
        return decoded_preds, decoded_labels, generated_tokens, generated_tokens_mask
    else:
        return decoded_preds, decoded_labels
