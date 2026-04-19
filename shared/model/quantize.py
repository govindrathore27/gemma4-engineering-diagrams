# shared/model/quantize.py
"""Export a LoRA adapter to GGUF 4-bit for offline (llama.cpp) use."""
from pathlib import Path

SUPPORTED_QUANT_TYPES = ["q4_k_m", "q8_0", "f16"]


def export_gguf(adapter_dir: str, output_path: str, quant_type: str = "q4_k_m") -> None:
    """Export a trained LoRA adapter to GGUF format for offline deployment.

    Args:
        adapter_dir: Path to directory containing the trained LoRA adapter.
        output_path: Path where the GGUF file will be saved.
        quant_type: Quantization method. One of: q4_k_m, q8_0, f16. Defaults to q4_k_m.

    Raises:
        ValueError: If adapter_dir, output_path are empty, or quant_type is unsupported.
        FileNotFoundError: If adapter_dir does not exist.
    """
    # Validate inputs
    if not adapter_dir or not adapter_dir.strip():
        raise ValueError("adapter_dir must be a non-empty string")
    if not output_path or not output_path.strip():
        raise ValueError("output_path must be a non-empty string")
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"quant_type '{quant_type}' is not supported. "
            f"Choose from: {', '.join(SUPPORTED_QUANT_TYPES)}"
        )

    # Validate adapter directory exists
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir!r}. "
            "Ensure the training step completed and the adapter was saved."
        )

    # Create output directory
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter from {adapter_dir}...")
    print(f"Exporting to GGUF with quantization: {quant_type}")

    # Defer heavy imports until inside function
    from unsloth import FastLanguageModel

    # Load the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=4096,
        load_in_4bit=True,
    )

    # Export to GGUF
    model.save_pretrained_gguf(
        str(out.parent),
        tokenizer,
        quantization_method=quant_type,
    )
    print(f"GGUF exported to {out.parent}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Export a LoRA adapter to GGUF format for offline (llama.cpp) use."
    )
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--output", required=True, help="Output GGUF file path")
    p.add_argument(
        "--quant",
        default="q4_k_m",
        choices=SUPPORTED_QUANT_TYPES,
        help="Quantization method (default: q4_k_m)",
    )
    args = p.parse_args()
    export_gguf(args.adapter, args.output, args.quant)
