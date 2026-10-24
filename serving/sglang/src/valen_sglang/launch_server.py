"""Launch SGLang with the Valen plugin."""

import os
import sys


def main():
    plugins = os.environ.get("SGLANG_PLUGINS", "").split(",")
    os.environ["SGLANG_PLUGINS"] = ",".join(
        dict.fromkeys(filter(None, [*plugins, "valen"]))
    )
    from sglang.srt.plugins import load_plugins

    load_plugins()
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
