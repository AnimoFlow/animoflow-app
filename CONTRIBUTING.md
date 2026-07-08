# Contributing to animoflow-app

This repository is the **deployment wrapper** for the hosted AnimoFlow demo
(the Hugging Face Space). Most contributions belong in the components it
wraps:

- API surface, job contract, error copy → [animoflow-api](https://github.com/AnimoFlow/animoflow-api)
- Models, pipeline nodes, retargeting → [comfyui-animoflow](https://github.com/AnimoFlow/comfyui-animoflow)
- Blender addon → [animoflow-blender](https://github.com/AnimoFlow/animoflow-blender)

If your change really is about the Space deployment itself (bootstrap,
ZeroGPU handling, the Gradio wrapper): external contributions require a
one-time Contributor License Agreement — a bot will ask on your first PR.
CLA text and the per-repo license map: [AnimoFlow/legal](https://github.com/AnimoFlow/legal).

## Development

```bash
pip install pytest
python -m pytest -q     # unit tests; no GPU or Space needed
```

Note that a real end-to-end run requires the Space environment (ZeroGPU,
private checkpoint repos) — deployment changes are verified on the Space
itself by the maintainer.

Questions first? Open an issue or write to guy@animoflow.ai.
