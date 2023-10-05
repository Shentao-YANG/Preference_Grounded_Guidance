import argparse
from transformers import SchedulerType, MODEL_MAPPING
import torch

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="An optional input test data file to evaluate the metrics (rouge, meteor, ...) on (a jsonlines or csv file)."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. "
            "Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after tokenization. "
            "Sequences longer than this will be truncated, sequences shorter will be padded."
            "During ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation target text after tokenization. "
            "Sequences longer than this will be truncated, sequences shorter will be padded. "
            "Will default to `max_target_length`. This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    ########### set the following three for the debugging mode ###########
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.",
    )
    ######################################################################
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="t5-small",
    )
    parser.add_argument(
        "--rew_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="t5-small",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).",
    )
    parser.add_argument(
        "--resize_position_embeddings",
        type=bool,
        default=None,
        help="Whether to automatically resize the position embeddings if `max_source_length` exceeds the model's position embeddings.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before perform a backward/update pass.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible training.")
    parser.add_argument(
        "--model_type", type=str, default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    ############################################################################################################
    parser.add_argument(
        "--device", type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="Device for model training"
    )
    parser.add_argument("--use_softplus", type=int, default=1,
                        help="Whether use `softplus` or `sigmoid` in bounding the output of the reward model")
    parser.add_argument("--gradient_clip", type=int, default=1)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument("--rew_num_train_epochs", type=float, default=0.1)
    parser.add_argument("--rew_learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_decay", type=float, default=0.8)
    parser.add_argument("--weight_decay_count", type=int, default=2)
    parser.add_argument("--early_stop_count", type=int, default=7)
    parser.add_argument("--reward_learning_batch_size", type=int, default=8)
    parser.add_argument("--rew_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--rew_eval_period", type=float, default=0.25)       # fraction of a epoch
    parser.add_argument("--policy_eval_period", type=float, default=1.0)

    parser.add_argument("--rew_checkpoint_path", type=str, default=None)
    parser.add_argument("--policy_checkpoint_path", type=str, default=None)

    parser.add_argument("--reward_learning_samples", type=int, default=3)
    parser.add_argument("--agg_func", type=str, default="avg")
    parser.add_argument("--soft_maxmin_temp", type=float, default=2.)
    parser.add_argument("--reinforce_coeff", type=float, default=-1.)       # default: "-1" suppress this loss component
    parser.add_argument("--max_entropy_coeff", type=float, default=-1.)     # default: "-1" suppress this loss component
    parser.add_argument("--num_reinforce_samples", type=int, default=1)
    parser.add_argument("--use_q_for_weight", type=int, default=0)      # default: use r(s,a) for weight
    parser.add_argument("--reward_retrain_period", type=float, default=0.5)
    parser.add_argument("--exp_in_wmle", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use in the optimizer.")
    parser.add_argument("--reset_optim", type=int, default=0)
    parser.add_argument("--expid", type=str, default="1", help="Experiment ID to organize results/logs.")
    ############################################################################################################
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    if args.output_dir is None:
        args.output_dir = f"./output/Exp{args.expid}/seed{args.seed}"

    return args
