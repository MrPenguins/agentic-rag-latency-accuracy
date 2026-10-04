# Local Agentic RAG: Latency–Accuracy Trade-off

Code and experimental results for **Quantifying the Latency-Accuracy Trade-off
in Local Agentic RAG Workflows**, accepted at the SLM-Agents Workshop,
NeurIPS 2026.

**Haoqi Zhang, Niila Siilasjoki, Mikko Raatikainen, and Jukka K. Nurminen**  
Department of Computer Science, University of Helsinki

We compare five pipelines (Baseline, Linear V1/V2, and Corrective V1/V2)
across Llama 3.1 8B, Qwen 3.5 9B, and Qwen 3.5 4B on 500 HotpotQA
questions. Retrieval uses each question's ten candidate paragraphs;
the generator receives two documents.

## Files

- `src/local_agent_rag/`: pipelines, prompts, retrieval, and evaluation.
- `scripts/`: data preparation, experiments, judging, and result analysis.
- `configs/`: settings for the three models.
- `data/`: the fixed 500-question subset.
- `results/raw/` and `results/judged/`: original experiment outputs and judge scores.
- `tests/`: existing offline tests.

## Reproduce the paper's results

Use Python 3.12 in a virtual environment. Run commands from the repository root
with that environment's Python. The following uses the included CSVs and needs
no GPU, model downloads, or API key:

```sh
python -m pip install "matplotlib>=3.10,<4" "Pillow>=11,<13"
python scripts/reproduce_results.py --verify
```

This writes the table, McNemar comparisons, metrics, and efficiency-frontier
plot to `results/summary/`. `--verify` checks agreement with the paper's values.

## Run model experiments

Install the full dependencies and start Ollama with the required models.
GPU-enabled PyTorch must match your GPU and driver.

```sh
python -m pip install -r requirements.txt
ollama pull llama3.1
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
python scripts/prepare_data.py --config configs/llama3.1_8b.yaml
python scripts/run_benchmark.py --config configs/llama3.1_8b.yaml --output artifacts/raw_llama3.1_8b.csv
```

Data preparation downloads HotpotQA and builds the shared retrieval store.
To run Qwen, use `configs/qwen3.5_9b.yaml` or `configs/qwen3.5_4b.yaml`
and a different output filename. Runs resume by skipping recorded question IDs.
Keep new outputs under `artifacts/`.

For external judging, set `DEEPSEEK_API_KEY` in your environment and run:

```sh
python scripts/run_judge.py --input artifacts/raw_llama3.1_8b.csv --output artifacts/judged_llama3.1_8b.csv --judge-model deepseek-v4-flash
```

Use a model identifier available from the provider. Historical judging used
`deepseek-chat` on June 4–6, 2026, recorded as DeepSeek-V4-Flash in non-thinking
mode. API judging is separate from the measured local latency.

After installing the full dependencies, run the tests with:

```sh
python -m unittest discover -s tests -v
```

## Experiment notes

- The reported setup used Windows, Ollama 0.30.0, CUDA 13.0, and an RTX 5060
  Laptop GPU (8 GB), with an Intel Core Ultra 7 255H and 32 GB RAM.
- Models used Q4_K_M quantization and temperature 0, with Qwen reasoning
  disabled. Retrieval combines BGE embeddings and BM25 with equal RRF weights.
  Corrective pipelines allow at most three generation–review passes.
- Match raw and judged CSVs by `question_id`; their row order can differ.
  CSV prefixes `correction_v1/v2` refer to Corrective V1/V2.
- The original sampling seed and model digests were not retained. The exact
  subset and recorded outputs are included; fresh runs can differ in answers
  and latency. Timings are from one execution per question and configuration.

## Citation

```bibtex
@misc{zhang2026localagenticrag,
  title = {Quantifying the Latency-Accuracy Trade-off in Local Agentic RAG Workflows},
  author = {Zhang, Haoqi and Siilasjoki, Niila and Raatikainen, Mikko and Nurminen, Jukka K.},
  year = {2026},
  howpublished = {SLMs for Agentic Systems (SLM-Agents) Workshop at NeurIPS 2026}
}
```

## Acknowledgments

This work was partly funded by local authorities (“Business Finland”) under
grant agreement 23004 ELFMo of the ITEA4 programme.

## License

Original code and documentation use the [MIT License](LICENSE).
HotpotQA data retain CC BY-SA 4.0; see [data attribution](THIRD_PARTY_ASSETS.md).
