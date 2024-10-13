from modal import App, web_endpoint, Image, method, enter, Mount


app = App("auto-ed-coder")


MODEL_NAME_1 = "bert-base-uncased"
MODEL_NAME_2 = "joetey/bert-base-uncased-finetuned-set_3"


def download_models_to_folders():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    import os

    print("Starting model downloads...")
    os.makedirs(MODEL_NAME_1, exist_ok=True)
    os.makedirs(MODEL_NAME_2, exist_ok=True)
    print(f"Created directories: {MODEL_NAME_1}, {MODEL_NAME_2}")

    for model_name in [MODEL_NAME_1, MODEL_NAME_2]:
        snapshot_download(
            model_name,
            local_dir=model_name,
            ignore_patterns=["*.pt"],
        )
        print(f"Model {model_name} downloaded successfully")

    move_cache()
    print("Cache moved")


image = (
    Image.debian_slim()
    .pip_install("transformers", "torch", "huggingface_hub", "nltk", "pydantic")
    .run_function(download_models_to_folders)
)


@app.cls(gpu="any", image=image)
class HFEngine:
    @enter()
    def load_models(self):
        from transformers import AutoTokenizer, AutoModel
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME_1)
        self.model1 = AutoModel.from_pretrained(MODEL_NAME_1).to(self.device)
        self.tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME_2)
        self.model2 = AutoModel.from_pretrained(MODEL_NAME_2).to(self.device)

    @method()
    def predict(self, text: str, MODEL_NAME: str):
        tokenizer = self.tokenizer1 if MODEL_NAME == MODEL_NAME_1 else self.tokenizer2
        model = self.model1 if MODEL_NAME == MODEL_NAME_1 else self.model2

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)
        import torch

        with torch.no_grad():
            outputs = model(**inputs)

        # Get the embedding from the last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return {"embedding": embedding.tolist()}


@app.function()
@web_endpoint(label="bert-base-uncased", method="POST")
def inference_endpoint_model1(input_data: dict):
    text = input_data.get("text", "")
    engine = HFEngine()
    result = engine.predict.remote(text, MODEL_NAME=MODEL_NAME_1)
    return result


@app.function()
@web_endpoint(label="bert-set-3-fine-tuned", method="POST")
def inference_endpoint_model2(input_data: dict):
    text = input_data.get("text", "")
    engine = HFEngine()
    result = engine.predict.remote(text, MODEL_NAME=MODEL_NAME_2)
    return result


@app.function(
    image=image, mounts=[Mount.from_local_dir("backend/sae/", remote_path="/root/sae")]
)
@web_endpoint(label="heatmap-generator", method="POST")
def generate_heatmap(input_data: dict):
    import torch
    from nltk.tokenize import sent_tokenize
    import nltk
    import pickle
    import json

    nltk.download("punkt")
    import os

    import torch.nn as nn
    import torch.nn.functional as F
    from pydantic import BaseModel
    from huggingface_hub import PyTorchModelHubMixin

    class SparseAutoencoderConfig(BaseModel):
        d_model: int
        d_sparse: int
        sparsity_alpha: float = 0.0  # doesn't matter for inference

    class SparseAutoencoder(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config: SparseAutoencoderConfig):
            super().__init__()
            self.config = config

            # from https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder
            self.enc_bias = nn.Parameter(torch.zeros(config.d_sparse))
            self.encoder = nn.Linear(config.d_model, config.d_sparse, bias=False)
            self.dec_bias = nn.Parameter(torch.zeros(config.d_model))
            self.decoder = nn.Linear(config.d_sparse, config.d_model, bias=False)

        def forward(
            self,
            x: torch.FloatTensor,
            return_loss: bool = False,
            sparsity_scale: float = 1.0,
            new_loss: bool = True,
        ):
            f = self.encode(x)
            y = self.decode(f)

            if return_loss:
                reconstruction_loss = F.mse_loss(y, x)
                # print(x.shape)

                decoder_norms = torch.norm(self.decoder.weight, dim=0)

                if new_loss:
                    sparsity_loss = (
                        sparsity_scale
                        * self.config.sparsity_alpha
                        * (f.abs() @ decoder_norms).sum()
                    )  # TODO: change this to the actual loss function
                else:
                    sparsity_loss = (
                        sparsity_scale * self.config.sparsity_alpha * (f.abs().sum())
                    )

                loss = reconstruction_loss / x.shape[1] + sparsity_loss
                return y, f, loss, reconstruction_loss

            return y, f, None, None

        def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
            return F.relu(self.encoder(x - self.dec_bias) + self.enc_bias)

        def decode(self, f: torch.FloatTensor) -> torch.FloatTensor:
            return self.decoder(f) + self.dec_bias

        def load(self, path: os.PathLike, device: torch.device = "cpu"):
            self.load_state_dict(torch.load(path, map_location=device))

    text = input_data.get("text", "")
    model_type = input_data.get("model_type", "pre-trained")  # 'base' or 'fine-tuned'
    first_n_features = input_data.get("first_n_features", 10)

    # Load the configuration
    if model_type == "fine-tuned":
        config_path = "/root/sae/fine-tuned/config.json"
    else:
        config_path = "/root/sae/pre-trained/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Load the pre-trained model from the pickle file
    sae_config = SparseAutoencoderConfig(
        d_model=config["dimensions"],
        d_sparse=8 * config["dimensions"],
        sparsity_alpha=config["sparsity_alpha"],
    )
    sae_model = SparseAutoencoder(sae_config)
    if model_type == "fine-tuned":
        model_path = "/root/sae/fine-tuned/sae.pkl"
    else:
        model_path = "/root/sae/pre-trained/sae.pkl"
    with open(model_path, "rb") as f:
        model_state_dict = pickle.load(f)
        sae_model.load_state_dict(model_state_dict)

    # 1. Split the text into sentences
    sentences = sent_tokenize(text)

    # 2. Embed each sentence
    engine = HFEngine()
    embeddings = []
    for sentence in sentences:
        if model_type == "fine-tuned":
            result = engine.predict.remote(sentence, MODEL_NAME=MODEL_NAME_2)
        else:
            result = engine.predict.remote(sentence, MODEL_NAME=MODEL_NAME_1)
        embedding = result["embedding"]
        embeddings.append(embedding)

    # 3. Get feature activations
    feature_activations = []
    for embedding in embeddings:
        embedding_tensor = torch.tensor(embedding)
        activation = sae_model.forward(embedding_tensor)[1]
        feature_activations.append(activation[:first_n_features].tolist())

    # 4. Prepare output
    output = []
    for sentence, embedding, activation in zip(
        sentences, embeddings, feature_activations
    ):
        output.append(
            {
                "sentence": sentence,
                "embedding": embedding,
                "feature_activations": activation,
            }
        )

    # 5. Return the output
    return output
