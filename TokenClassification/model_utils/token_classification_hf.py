# WORKING WITH TRANSFORMERS V4.5.X

from datetime import datetime
from model_utils.utils import run_bash_command
from shlex import quote
import os
import random


def run_token_classification_model(
    run_ner_file: str,
    train_file: str,
    validation_file: str,
    test_file: str,
    model_name: str,
    cache_dir: str,
    output_dir: str,
    num_train_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    fp16: bool,
    max_seq_length: int,
    label_all_tokens: bool = False,
    lr_scheduler_type: str = None,
    warmup_ratio: str = None,
    warmup_steps: str = None,
    experiment_name: str = str(datetime.now()),
    deepspeed: bool = False,
    deepspeed_gpu_id: int = 0,
) -> None:

    print(f"Logdir -> {os.path.abspath(os.path.join('runs',str(experiment_name)))}")
    print(f"Output path -> {os.path.abspath(output_dir)}")
    print(
        f"Params log -> {os.path.abspath(os.path.join(output_dir, 'tk_training_parameters.txt'))}"
    )
    print()

    if "luke" not in model_name:
        if deepspeed:
            header: str = (
                f"deepspeed --include localhost:{quote(str(deepspeed_gpu_id))} "
                f"{quote(run_ner_file)} "
                f"--deepspeed model_utils/ds_config_zero2.json"
            )
        else:
            header: str = f"python3 {quote(run_ner_file)}"

        command: str = (
            f"{header} "
            f"--task_name ner "
            f"--train_file {quote(train_file)} "
            f"--validation_file {quote(validation_file)} "
            f"--test_file {quote(test_file)} "
            f"--model_name_or_path {quote(model_name)} "
            f"--output_dir {quote(output_dir)} "
            f"--num_train_epochs {quote(str(num_train_epochs))} "
            f"--per_device_train_batch_size {quote(str(batch_size))} "
            f"--per_device_eval_batch_size {quote(str(batch_size))} "
            f"--gradient_accumulation_steps {quote(str(gradient_accumulation_steps))} "
            f"--learning_rate {quote(str(learning_rate))} "
            f"--seed {random.randint(1,1000)} "
            f"--preprocessing_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
            f"--dataloader_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
            f"--evaluation_strategy epoch "
            f"--logging_strategy epoch "
            f"--logging_dir {quote(os.path.join('runs',str(experiment_name)))} "
            f"--save_strategy epoch "
            f"--load_best_model_at_end "
            f"--metric_for_best_model eval_f1 "
            f"--greater_is_better True "
            f"--save_total_limit 1 "
            f"--do_train "
            f"--do_eval "
            f"--do_predict "
            f"--overwrite_output_dir "
            f"--overwrite_cache "
            f"--dataloader_pin_memory "
        )

        if fp16:
            command += " --fp16 "

        if label_all_tokens:
            command += " --label_all_tokens"

        if lr_scheduler_type:
            command += f"--lr_scheduler_type {quote(str(lr_scheduler_type))} "

        if warmup_ratio:
            command += f"--warmup_ratio {quote(str(warmup_ratio))} "

        if warmup_steps:
            command += f"--warmup_steps {quote(str(warmup_steps))} "
        if max_seq_length:
            command += f"--max_seq_length {quote(str(max_seq_length))} "
        with open(
            os.path.join(output_dir, "ner_training_parameters.txt"),
            "w+",
            encoding="utf-8",
        ) as filelog:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), file=filelog)
            print(command.replace("--", "\\\n--"), file=filelog)
    else:
        print("LUKE MODEL DETECTED")
        command: str = (
            f"accelerate launch --mixed_precision fp16 third_party/run_luke_ner_no_trainer.py "
            f"--task_name ner "
            f"--train_file {quote(train_file)} "
            f"--validation_file {quote(validation_file)} "
            f"--test_file {quote(test_file)} "
            f"--model_name_or_path {quote(model_name)} "
            f"--output_dir {quote(output_dir)} "
            f"--num_train_epochs {quote(str(num_train_epochs))} "
            f"--per_device_train_batch_size {quote(str(batch_size))} "
            f"--per_device_eval_batch_size {quote(str(batch_size))} "
            f"--gradient_accumulation_steps {quote(str(gradient_accumulation_steps))} "
            f"--learning_rate {quote(str(learning_rate))} "
            f"--seed {random.randint(1, 1000)} "
        )

    run_bash_command(command)


def train_token_classification_model(
    run_ner_file: str,
    train_file: str,
    validation_file,
    model_name: str,
    cache_dir: str,
    output_dir: str,
    num_train_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    fp16: bool,
    label_all_tokens: bool = False,
    deepspeed: bool = False,
    deepspeed_gpu_id: int = 0,
) -> None:

    if deepspeed:
        header: str = (
            f"deepspeed --include localhost:{quote(str(deepspeed_gpu_id))} "
            f"{quote(run_ner_file)} "
            f"--deepspeed model_utils/ds_config_zero2.json"
        )
    else:
        header: str = f"python3 {quote(run_ner_file)}"

    command: str = (
        f"{header} "
        f"--task_name ner "
        f"--train_file {quote(train_file)} "
        f"--validation_file {quote(validation_file)} "
        f"--model_name_or_path {quote(model_name)} "
        f"--cache_dir {quote(cache_dir)} "
        f"--output_dir {quote(output_dir)} "
        f"--num_train_epochs {quote(str(num_train_epochs))} "
        f"--per_device_train_batch_size {quote(str(batch_size))} "
        f"--per_device_eval_batch_size {quote(str(batch_size))} "
        f"--gradient_accumulation_steps {quote(str(gradient_accumulation_steps))} "
        f"--learning_rate {quote(str(learning_rate))} "
        f"--seed {random.randint(1,1000)} "
        f"--preprocessing_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
        f"--dataloader_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
        f"--evaluation_strategy no "
        f"--logging_strategy epoch "
        f"--save_strategy no "
        # f"--load_best_model_at_end "
        # f"--metric_for_best_model eval_f1 "
        # f"--greater_is_better True "
        f"--do_train "
        f"--do_eval "
        f"--overwrite_output_dir "
        f"--overwrite_cache "
        f"--dataloader_pin_memory "
    )

    if fp16:
        command += " --fp16"

    if label_all_tokens:
        command += " --label_all_tokens"

    with open(
        os.path.join(output_dir, "ner_training_parameters.txt"), "w+", encoding="utf-8"
    ) as filelog:
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), file=filelog)
        print(command.replace("--", "\\\n--"), file=filelog)

    run_bash_command(command)


def eval_token_classification_model(
    run_ner_file: str,
    train_file: str,
    validation_file: str,
    model_name: str,
    output_dir: str,
    fp16: bool,
    batch_size: int,
    label_all_tokens: bool = False,
    deepspeed: bool = False,
    deepspeed_gpu_id: int = 0,
) -> None:

    if deepspeed:
        header: str = (
            f"deepspeed --include localhost:{quote(str(deepspeed_gpu_id))} "
            f"{quote(run_ner_file)} "
            f"--deepspeed model_utils/ds_config_zero2.json"
        )
    else:
        header: str = f"python3 {quote(run_ner_file)}"

    command: str = (
        f"{header} "
        f"--task_name ner "
        f"--train_file {quote(train_file)} "
        f"--validation_file {quote(validation_file)} "
        f"--model_name_or_path {quote(model_name)} "
        f"--cache_dir {quote(model_name)} "
        f"--output_dir {quote(output_dir)} "
        f"--per_device_eval_batch_size {batch_size} "
        f"--preprocessing_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
        f"--dataloader_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
        f"--do_eval "
        f"--overwrite_cache "
        f"--overwrite_output_dir "
        f"--logging_strategy no "
        f"--save_strategy no "
        f"--evaluation_strategy no "
        f"--dataloader_pin_memory "
    )

    if fp16:
        command += " --fp16"

    if label_all_tokens:
        command += " --label_all_tokens"

    run_bash_command(command)


def inference_token_classification_model(
    run_ner_file: str,
    train_file: str,
    test_file: str,
    model_name: str,
    output_dir: str,
    fp16: bool,
    batch_size: int,
    label_all_tokens: bool = False,
    deepspeed: bool = False,
    deepspeed_gpu_id: int = 0,
) -> None:

    if deepspeed:
        header: str = (
            f"deepspeed --include localhost:{quote(str(deepspeed_gpu_id))} "
            f"{quote(run_ner_file)} "
            f"--deepspeed model_utils/ds_config_zero2.json"
        )
    else:
        header: str = f"python3 {quote(run_ner_file)}"

    command: str = (
        f"{header} "
        f"--task_name ner "
        f"--train_file {quote(train_file)} "
        f"--validation_file {quote(test_file)} "
        f"--test_file {quote(test_file)} "
        f"--model_name_or_path {quote(model_name)} "
        f"--cache_dir {quote(model_name)} "
        f"--output_dir {quote(output_dir)} "
        f"--per_device_eval_batch_size {batch_size} "
        f"--preprocessing_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
        f"--dataloader_num_workers {quote(str(os.cpu_count()//4)) if os.cpu_count()>16 else quote(str(os.cpu_count()))} "
        f"--do_predict "
        f"--overwrite_cache "
        f"--overwrite_output_dir "
        f"--logging_strategy no "
        f"--save_strategy no "
        f"--evaluation_strategy no "
        f"--dataloader_pin_memory "
    )

    if fp16:
        command += " --fp16"

    if label_all_tokens:
        command += " --label_all_tokens"

    run_bash_command(command)
