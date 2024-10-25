def run_sample(gpt_model, sae, sample, device="cuda:0", mlp_layer = 5):
  logits, activations = gpt_model.run_with_cache(sample)

  mlp_out = activations.cache_dict[f"blocks.{mlp_layer}.hook_mlp_out"].to(device) # example MLP output, shape: (1, # samples, # dim)

  y, f, loss, reconstruction_loss = sae(mlp_out, True)
  feature_activations = sae.feature_activations(mlp_out)

  return {
    "mlp_out": mlp_out,
    "feature_activations": feature_activations,
    "f": f,
    "loss": loss,
    "reconstruction_loss": reconstruction_loss,
  }