import torch


def save_weights_with_description(weights, description, path):
    torch.save({"weights": weights, "description": description}, path)


def load_weights_with_description(path, print_description=True):
    saved_weights = torch.load(path)
    assert "weights" in saved_weights and "description" in saved_weights

    if print_description:
        print(f"Loaded weights with description: {saved_weights['description']}")

    return saved_weights["weights"]
