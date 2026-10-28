# Backend validation tooling

Complete application tutorials live only in the [official Cookbook](https://valen-team.github.io/cookbook/) ([简体中文](https://valen-team.github.io/zh-Hans/cookbook/)). The [version manifest](https://valen-team.github.io/cookbook/versions.json), [capability matrix](https://valen-team.github.io/cookbook/capabilities.json), [performance and limits](https://valen-team.github.io/cookbook/validation/) and [migration map](https://valen-team.github.io/cookbook/migration/) are shared references. API docs own protocols/fields/errors/minimal calls; deployment docs own installation/startup/backend configuration; Cookbook owns complete workflows.

## Fast checks: no models

The canonical protocol fixture is `tests/fixtures/backend-contract.json`. It covers choice, predicate/noul, score, string/boolean values including collisions, candidate order, two scalar temperatures and invalid input. Run the synthetic request/response adapter check using existing SDK source checkouts and already installed JS dependencies:

```sh
python scripts/validation/check_protocols.py \
  --python-sdk-dir ../valen-sdk-python \
  --javascript-sdk-dir ../valen-sdk-js \
  --output synthetic-adapters.json
```

`Backend protocol contract` CI checks the main/serving Python adapters plus pinned stable Python and JavaScript SDK sources. Its constructed responses are explicitly labeled `synthetic_adapter`: this is not real inference or model quality. No synthetic model was used for P1.

The website's `npm run test:cookbook` checks business review policies, direct code imports, and hashes binding the published code/sample files to real execution. `npm run build` builds both languages; `npm run check-links` checks generated files and anchors. The download ZIP is generated from the same website source files. Code/sample changes require fresh real execution rather than editing the stored hashes to bypass validation.

## Real checks: cluster caches only

Use an existing compute allocation and cached qualified weights. Read the actual model/runtime locations from the run records; never assume another user's cache paths. Do not download or relay weights, images, toolchains or runtime archives through a metered local computer. Submit bounded jobs that keep running if SSH disconnects; inspect existing job/output state before retrying. Preserve failed or interrupted reports.

Start the intended backend with its documented model name and explicit CPU/device settings. Then run the shared fixtures against its HTTP endpoint:

```sh
python scripts/validation/check_backends.py \
  --base-url http://127.0.0.1:8000/v1 \
  --image /existing/cache/gradient.png \
  --video /existing/cache/colors.mp4 \
  --output backend-contract.json
```

Omit `--video` for prebuilt GGUF runtime 0.1.0: direct video is not compiled into that package. Its failed P1 probe remains in the published evidence. Source-build video validation is separately named and never substitutes for prebuilt qualification. Native Python video in the localhost benchmark adapter is explicitly materialized as a local file; it does not add video support to production SGLang/vLLM plugins.

Check the actual installed Python SDK and registry-verified JavaScript package too:

```sh
python scripts/validation/check_sdk_http.py \
  --base-url http://127.0.0.1:8000/v1 \
  --image /existing/cache/gradient.png \
  --video /existing/cache/colors.mp4 \
  --output python-sdk-http.json
```

`check_sdk_http.mjs` reads the same prepared fixture payloads and expected real responses on stdin; `check_backends.prepared_cases` supplies them. The P1 cluster follow-up executed it against the cached, SHA256-verified **@valen-team/sdk@0.2.0** npm archive. The unmodified old `@valen/sdk` package name was not used as the published identity.

Execute all five canonical Cookbook scripts from the website repository. Preserve stdout, stderr, exit status, model revision, dependency/runtime versions and code hashes. A successful import, mock response, synthetic model or graph load cannot replace this step. `Valen.local` is an ONNX path; connect to GGUF through `Valen.remote`.

## Reproducible latency and RSS

`benchmark_backends.py` takes a JSON config with existing server commands, environment overrides, model provenance, input paths, media budgets and runtime versions. The exact P1 config is downloadable with the [performance evidence](https://valen-team.github.io/cookbook/validation/). Its paths record the original cluster caches; substitute your established paths when reproducing elsewhere.

```sh
python scripts/validation/benchmark_backends.py \
  --config /existing/run/runtime-config.json --output /existing/run/results \
  --warmup 2 --runs 7
```

One backend/modality per fresh process; four threads; one cold request; two extra warmups; seven sequential measurements; no concurrency-throughput claim. Startup includes process imports and readiness. OS page caches are not cleared. The native/ONNX localhost adapter records `predict` wall time separately from client HTTP time; GGUF server inference time remains unavailable. RSS is Linux server-process `VmHWM`, not model file bytes or GPU VRAM. Same-worker rows must keep the hardware/allocation and inputs fixed. `--resume` refuses a different worker/hardware and preserves existing rows; never stitch different machines into one performance comparison.

P1 measured 0.8B native FP32, ONNX FP32, and the exact prebuilt GGUF Q8_0 text/fixed-image paths. Native/ONNX direct video is measured in that same allocation. The video-enabled GGUF source build was checked in a separate follow-up allocation; no timing from it is pooled into the comparison. 2B/4B, GPU performance, WebGPU local execution and production concurrency were not rerun.

## Release gate

Before a relevant model/runtime/SDK release:

1. Freeze intended model revisions, weights/manifest hashes, SDK versions and runtime source/binary hashes. Keep both protocols compatible; document real capability differences.
2. Run the fast checks, then the real fixed-fixture/SDK/Cookbook checks on cached assets. For quantized/numerical comparisons, retain the prior fixed gates and the native baseline; do not retune gates after seeing results.
3. Publish immutable small JSON evidence with source hashes, successful scopes, failures and explicit unexecuted scopes. A supported check that failed or did not run cannot be marked passed. Unsupported prebuilt video remains excluded from positive qualification and visible as a failed probe.
4. Verify the actual remote release, site build/links, published package identities, and HF README/checksum changes using server metadata. Do not redownload large assets just to verify publication. Never move an existing tag or replace an existing PyPI/npm version.

P1 keeps Python `valen-sdk==0.2.0`, JavaScript `@valen-team/sdk@0.2.0` / `npm-v0.2.0`, and runtime `gguf-runtime-v0.1.0` unchanged. Existing calls passed; no SDK inference fix or new registry version was needed. A future SDK code fix must use a new version and its safe publishing workflow.
