import argparse

try:
    import fire
except ModuleNotFoundError:
    fire = None

from cka_evaluate import DEFAULT_TEST_FILE, inference as cka_inference


def inference(
    batch_size: int = 4,
    resume_from_checkpoint_with_engram: str = "",
    resume_from_checkpoint_without_engram: str = "",
    base_model: str = "Qwen3-1.7B",
    test_file: str = DEFAULT_TEST_FILE,
    cutoff_len: int = 2048,
    engram_layer_ids: str = "6,13,20",
    engram_float32: bool = False,
    k: int = 5,
    max_samples: int = 0,
    load_device: str = "auto",
    result_json: str = "",
    plot_path: str = "",
):
    return cka_inference(
        batch_size=batch_size,
        resume_from_checkpoint_with_engram=resume_from_checkpoint_with_engram,
        resume_from_checkpoint_without_engram=resume_from_checkpoint_without_engram,
        base_model=base_model,
        test_file=test_file,
        cutoff_len=cutoff_len,
        engram_layer_ids=engram_layer_ids,
        engram_float32=engram_float32,
        k=k,
        max_samples=max_samples,
        hidden_state_target="sequence_last",
        load_device=load_device,
        result_json=result_json,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(inference)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--resume_from_checkpoint_with_engram", type=str, default="")
        parser.add_argument("--resume_from_checkpoint_without_engram", type=str, default="")
        parser.add_argument("--base_model", type=str, default="Qwen3-1.7B")
        parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
        parser.add_argument("--cutoff_len", type=int, default=2048)
        parser.add_argument("--engram_layer_ids", type=str, default="6,13,20")
        parser.add_argument("--engram_float32", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--k", type=int, default=5)
        parser.add_argument("--max_samples", type=int, default=0)
        parser.add_argument("--load_device", type=str, default="auto")
        parser.add_argument("--result_json", type=str, default="")
        parser.add_argument("--plot_path", type=str, default="")
        args = parser.parse_args()
        inference(**vars(args))
