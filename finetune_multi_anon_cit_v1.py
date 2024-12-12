from finetune_main import train_model, get_dataset_size
from gliner.training import TrainingArguments
import torch
import os

if __name__ == "__main__":
    base_model_name = "gliner_multi-v2.1"
    base_model = f"urchade/{base_model_name}"
    custom_model_name = f"anon_cit_v1_{base_model_name}"
    model_dir = f"models_generalist/{custom_model_name}"
    data_dir = f"generalist_data/{custom_model_name}_data"
    train_path = f"finetune_data/merged_v2.json"

    print(f":::: Finetuning on dataset {train_path}")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # calculate number of epochs
    num_steps = 30000
    batch_size = 8
    data_size = get_dataset_size(train_path)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    print(f"number of epochs : {num_epochs}")

    split_ratio = 0.9
    learning_rate = 5e-6
    weight_decay = 0.05
    compile_model = False

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    training_args = TrainingArguments(
    output_dir=model_dir,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    others_lr=learning_rate,
    others_weight_decay=weight_decay,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    evaluation_strategy="steps",
    eval_steps=3000,
    save_steps=3000,
    save_total_limit=5,
    dataloader_num_workers=8,
    use_cpu=(device == torch.device('cpu')),
    logging_first_step=True,
    logging_steps=6000,
    report_to="wandb",
    )

    train_model(base_model, 
                model_dir, 
                data_dir,
                train_path, 
                split_ratio, 
                compile_model,
                training_args)