"""Launch native vLLM pooling with the Valen plugin."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Valen-Team/Valen-2B")
    parser.add_argument("--served-model-name", default="valen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--revision")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    args = parser.parse_args()
    plugins = os.environ.get("VLLM_PLUGINS", "valen").split(",")
    os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*plugins, "valen"]))
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from valen_vllm import register

    register()
    command = [
        "vllm",
        "serve",
        args.model_path,
        "--runner",
        "pooling",
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--limit-mm-per-prompt",
        '{"image":16,"video":0}',
    ]
    if args.revision:
        command += ["--revision", args.revision]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
