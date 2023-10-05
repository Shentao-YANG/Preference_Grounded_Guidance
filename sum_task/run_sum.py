import json
import os
import random
from undecorated import undecorated
from types import MethodType
import datasets
import nltk
import torch
from torch.utils.data import DataLoader

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    T5Model
)
from transformers.utils import is_offline_mode

from parse_args import parse_args
from utils import (
    set_random_seed,
    prints,
    print_banner,
    load_raw_datasets,
    load_pretrained_model_config,
    load_pretrained_tokenizer,
    load_pretrained_model,
    preprocessing_raw_dataset,
    get_decoded_preds_labels_from_batch,
    print_dict
)
from reward_model import RewardModel, RewardModelTrainer
from policy_trainer import RewardPolicyTrainer


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    args = parse_args()

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if args.source_prefix is None and args.model_name_or_path in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        prints(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `", warning=True
        )

    if args.seed is not None:
        print_banner(f"Set random seed: {args.seed} !!!")
        set_random_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    raw_datasets = load_raw_datasets(args)

    config = load_pretrained_model_config(args, AutoConfig)
    tokenizer = load_pretrained_tokenizer(args, AutoTokenizer)
    model = load_pretrained_model(args, AutoModelForSeq2SeqLM, config, tokenizer)
    reward_transformer = load_pretrained_model(args, T5Model, config, tokenizer, reward_model=True)
    generate_with_grad = undecorated(model.generate)
    model.generate_with_grad = MethodType(generate_with_grad, model)

    metric = evaluate.load("rouge")     # metric for simulating the seq-level preference

    prefix = args.source_prefix if args.source_prefix is not None else ""
    column_names, text_column, summary_column, raw_datasets = preprocessing_raw_dataset(args, raw_datasets)

    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    prints(f'len(train_dataset) = {len(train_dataset)}')
    prints(f'len(eval_dataset) = {len(eval_dataset)}')
    prints(f'len(test_dataset) = {len(test_dataset)}')

    for index in random.sample(range(len(train_dataset)), 1):
        prints(f"Sample {index} of the training set: {train_dataset[index]}")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # data loaders for training the reward function
    rew_train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.reward_learning_batch_size
    )
    rew_eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.reward_learning_batch_size)

    # initialize the trainers
    reward_model = RewardModel(args=args, reward_transformer=reward_transformer)
    reward_model_trainer = RewardModelTrainer(
        args=args, reward_model=reward_model, policy=model, tokenizer=tokenizer, metric=metric,
        rew_train_dataloader=rew_train_dataloader, rew_eval_dataloader=rew_eval_dataloader
    )
    reward_policy_trainer = RewardPolicyTrainer(
        args=args, policy=model, reward_trainer=reward_model_trainer, tokenizer=tokenizer, metric=metric,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader
    )

    reward_policy_trainer.train()

    # evaluate the trained policy
    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }

    # create a new metric object for final evaluation
    metric = evaluate.load("rouge")

    def get_evaluation_results(dataloader):
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                decoded_preds, decoded_labels = get_decoded_preds_labels_from_batch(
                    args, batch, model, tokenizer, greedy_decoding=True,
                    num_samples=1, gen_kwargs=gen_kwargs, for_reward_training=False
                )
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        return result

    prints(f"********** Test Results **********")
    result = get_evaluation_results(test_dataloader)
    print_dict(result)
    all_results = {f"test_{k}": v for k, v in result.items()}
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f)


if __name__ == "__main__":
    main()
