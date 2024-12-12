from huggingface_hub import HfApi
import fire 

def push_to_hub(model_dir: str, output_id: str, output_branch: str = 'main'):
    """
    Pushes the model to the hub

    Parameters:
    model_dir (str): The directory of the model
    output_id (str): The Hugging Face model ID (AI-MO/Model-Name)
    """

    api = HfApi()
    api.upload_large_folder(folder_path = model_dir, 
                            repo_id = output_id,
                            repo_type = 'model',
                            revision = output_branch,
                            private = True)

if __name__ == "__main__":
    """
    Example usage:
    python push_to_hub.py \
    --model_dir='models_citation/CitationExtractorV0_gliner-multitask-large-v0.5' \
    --output_id='mertunsal/CitationExtractorV0_gliner-multitask-large-v0.5' \
    --output_branch='main'
    """
    fire.Fire(push_to_hub)