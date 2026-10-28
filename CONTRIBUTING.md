# Contributing to Valen

This guide is for people and coding agents. Contributions to code, documentation, evaluations and examples are welcome.

## Choose the right repository

- Model, training, evaluation and inference code: this repository.
- Website documentation and Cookbook: [valen-team.github.io](https://github.com/Valen-Team/valen-team.github.io).
- SDKs: [Python](https://github.com/Valen-Team/valen-sdk-python), [JavaScript](https://github.com/Valen-Team/valen-sdk-js), or [C++](https://github.com/Valen-Team/valen-sdk-cpp).

For a bug, open an [issue](https://github.com/Valen-Team/Valen/issues) with the expected and actual behavior, steps to reproduce, and relevant versions. For a larger change, discuss the approach in an issue before investing in implementation. For security reports, see [SECURITY.md](SECURITY.md).

## Make a change

1. Fork the relevant repository and create a branch.
2. Read the relevant documentation and existing implementation. Keep the change focused and preserve unrelated work.
3. Add or update tests when they help verify a behavior change. Keep English and Chinese documentation consistent when changing shared content.
4. Run the checks appropriate to the change and open a pull request. Describe the problem, the change, how it was verified, and any checks you could not run.

Agents should follow the same workflow, inspect repository instructions where present, and report what they actually verified. Do not include credentials, private data, model weights, or generated outputs in a contribution.

## Local checks

The project requires Python 3.10 or newer. For code changes, set up an appropriate environment and install the project and test dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The installation includes PyTorch and other substantial dependencies. Model, GPU, or optional-dependency tests may require additional setup; see [scripts/README.md](scripts/README.md) and the relevant evaluation or serving documentation. For a focused change, run the relevant tests first. Documentation-only changes can be checked with `git diff --check` and a review of links and examples; they do not require downloading models.

The backend protocol CI runs cross-repository adapter checks; see [.github/workflows/backend-contract.yml](.github/workflows/backend-contract.yml). If a check needs an environment you do not have, say so in the pull request.

## Releasing

Minor versions mark model releases; patch versions mark code or documentation changes without new weights. To cut a release:

1. In [CHANGELOG.md](CHANGELOG.md), move the `Unreleased` entries under a new version heading with today's date and update the compare links at the bottom.
2. Set the same version in `pyproject.toml` and in the citation block of both READMEs. Add a line to the News section and keep only the three most recent entries there.
3. Commit, then create an annotated tag and push it with the branch: `git tag -a vX.Y.Z -m "<summary>"` and `git push origin main vX.Y.Z`.
4. Create the GitHub Release from the tag, using that version's CHANGELOG section as the notes.

Never move or reuse an existing tag; a correction gets a new patch version.

Contributions are subject to the repository's [Apache 2.0 license](LICENSE); third-party models and datasets have their own licenses.
