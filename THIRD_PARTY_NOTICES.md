The file `src/ocr_service/paddle_pipeline.yaml` is derived from PaddleX,
Copyright (c) PaddlePaddle Authors, distributed under the Apache License 2.0.

Source: https://github.com/PaddlePaddle/PaddleX/blob/release/3.3/paddlex/configs/pipelines/PP-StructureV3.yaml

Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Model weights are downloaded from the official PaddleX model registry. Their
upstream licensing applies independently of this application's source code.
The local manifest records file hashes for subsequent integrity checks; these
are not independently published upstream checksums.

PyMuPDF/MuPDF are provided under AGPL or a commercial license from Artifex.
See https://pymupdf.readthedocs.io/en/latest/about.html.

`dots_prompt.txt` is from rednote-hilab/dots.ocr, MIT, Copyright (c) 2025
rednote-hilab. See LICENSES/dots-OCR-MIT.txt.
Source: https://github.com/rednote-hilab/dots.ocr/blob/master/dots_ocr/utils/prompts.py

`chandra_prompt.txt` is from datalab-to/chandra, Apache-2.0.
Source: https://github.com/datalab-to/chandra/blob/master/chandra/prompts.py
License: LICENSES/Chandra-code-Apache-2.0.txt.

The container builds llama.cpp b10000, MIT, Copyright (c) 2023-2026 The ggml
authors. Its license is copied into the image; see LICENSES/llama.cpp-MIT.txt.
Source: https://github.com/ggml-org/llama.cpp/tree/47a39665e7081dc482feec169961acc09750a5c4

Model licenses apply independently of the runtime and application:
- dots.mocr: MIT, https://huggingface.co/dots-studio/dots.mocr
- GLM-OCR: MIT, https://huggingface.co/zai-org/GLM-OCR
- LightOnOCR-3-1B: Apache-2.0, https://huggingface.co/lightonai/LightOnOCR-3-1B
- KDL-Frontier-Parser-nano: AGPL-3.0 as declared in its model card,
  https://huggingface.co/KDLAI/KDL-Frontier-Parser-nano
- Chandra OCR 2 weights: modified OpenRAIL-M. The author describes free research,
  personal use and startups below $2M funding/revenue; competitive API use is
  restricted and broader commercial use requires separate terms. Code Apache-2.0
  does not imply Apache licensing of weights.
  https://huggingface.co/datalab-to/chandra-ocr-2#commercial-usage

The GGUF conversions are third-party artifacts except GLM's ggml-org conversion;
their repositories, immutable revisions and available published SHA256 values
are recorded in src/ocr_service/vlm_models.py and result manifests.


## Additional OCR backends

- LightOnOCR-3-0.8B: Apache-2.0, [upstream](https://huggingface.co/lightonai/LightOnOCR-3-0.8B).
- Jina OCR v1: **CC-BY-NC-4.0**, [original model](https://huggingface.co/jinaai/jina-ocr-v1).
  The GGUF conversion does not remove the original noncommercial restriction.
  Its separate llama.cpp image modifies the DeepSeek RoPE assertion to accept configured theta.
- Surya OCR 2: code Apache-2.0; weights modified OpenRAIL, commercial conditions in
  [original card](https://huggingface.co/datalab-to/surya-ocr-2), including $5M funding/revenue threshold.
  Uses the official Datalab GGUF checkpoint, not an ONNX community export.
- Falcon-OCR v1.5: Apache-2.0, [upstream](https://huggingface.co/tiiuae/Falcon-OCR).
  Pinned source is downloaded with weights. At runtime its local writable copy is modified:
  CPU PyTorch attention/gate replace Triton kernels; unused GPU layout methods are removed.
  Original weights/source snapshot remain read-only. See docs/backends.md for changes.
- Nemotron 3.5 Parse: [OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3.5-Parse),
  tokenizer MIT with TOKENIZER_ATTRIBUTION.md. Snapshot preparation retains license,
  notices and attribution files. Uses official custom model and postprocessing code.
