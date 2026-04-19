from typing import Tuple, Any


def load_model(adapter_dir: str, base_model: str = "google/gemma-4-e4b-it") -> Tuple[Any, Any]:
    """Load a LoRA adapter from disk using unsloth.

    Args:
        adapter_dir: Path to the adapter directory. Must be non-empty.
        base_model: Base model name (default: "google/gemma-4-e4b-it")

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If adapter_dir is empty
    """
    if not adapter_dir or not adapter_dir.strip():
        raise ValueError("adapter_dir must be a non-empty string")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run(model: Any, tokenizer: Any, instruction: str, context: str = "") -> str:
    """Run a single inference with the model.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        instruction: The instruction/query to run
        context: Optional context information (default: "")

    Returns:
        The extracted model response
    """
    prompt = (
        f"<start_of_turn>user\n{instruction}\n{context}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract model response, with fallback to full string if delimiter not found
    if "<start_of_turn>model\n" in decoded:
        return decoded.split("<start_of_turn>model\n")[-1].strip()
    return decoded.strip()
