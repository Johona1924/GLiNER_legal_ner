from finetune import train_model
import os

if __name__ == "__main__":
    base_model_name = "gliner-multitask-large-v0.5"
    base_model = f"knowledgator/{base_model_name}"
    custom_model_name = f"CitationExtractorV0_{base_model_name}"
    model_dir = f"models_citation/{custom_model_name}"
    data_dir = f"citation_data/{custom_model_name}_data"
    train_path = f"citation_data/swiss_citation_extraction_filtered.json"
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    split_ratio = 0.9
    learning_rate = 5e-6
    weight_decay = 0.05
    batch_size = 2
    epochs = 1
    compile_model = False

    train_model(base_model, 
                model_dir, 
                data_dir,
                train_path, 
                split_ratio, 
                learning_rate, 
                weight_decay, 
                batch_size, 
                epochs, 
                compile_model)