# Third-party assets and licenses

This repository includes a HotpotQA subset and experiment outputs. Model
weights and third-party software are downloaded or installed separately.
The MIT license at the repository root applies to original code and
documentation, not to third-party assets.

| Asset | Use | Version or identifier | License / terms | Official source |
|---|---|---|---|---|
| HotpotQA | Evaluation questions and distractor contexts | `hotpot_dev_distractor_v1.json` | Dataset: CC BY-SA 4.0; upstream code: Apache-2.0 | https://hotpotqa.github.io/ and https://github.com/hotpotqa/hotpot |
| Meta Llama 3.1 8B | Local generator/planner/reviewer | Ollama tag `llama3.1`; historical digest not recorded | Llama 3.1 Community License and Acceptable Use Policy | https://github.com/meta-llama/llama-models/tree/main/models/llama3_1 |
| Qwen 3.5 9B | Local generator/planner/reviewer | Ollama tag `qwen3.5:9b`; historical digest not recorded | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-9B |
| Qwen 3.5 4B | Local generator/planner/reviewer | Ollama tag `qwen3.5:4b`; historical digest not recorded | Apache-2.0 | https://huggingface.co/Qwen/Qwen3.5-4B |
| BAAI/bge-base-en-v1.5 | Dense document embeddings | `BAAI/bge-base-en-v1.5` | MIT | https://huggingface.co/BAAI/bge-base-en-v1.5 |
| DeepSeek API | Post-hoc binary answer judge | Historical request alias `deepseek-chat`, 2026-06-04 to 2026-06-06 | Provider API terms; no model weights redistributed | https://api-docs.deepseek.com/updates/ |
| Ollama | Local model serving | 0.30.0 in reported experiments | MIT | https://github.com/ollama/ollama |
| LangChain / LangGraph | Prompt composition and agent routing | See `requirements.txt` | MIT | https://github.com/langchain-ai/langchain and https://github.com/langchain-ai/langgraph |
| ChromaDB | Persistent dense vector store | See `requirements.txt` | Apache-2.0 | https://github.com/chroma-core/chroma |
| rank-bm25 | Sparse BM25 retrieval | 0.2.2 | Apache-2.0 | https://github.com/dorianbrown/rank_bm25 |
| Matplotlib | Efficiency-frontier plotting | See `requirements.txt` | Matplotlib License | https://github.com/matplotlib/matplotlib |

## HotpotQA redistribution notice

`data/benchmark_subset_500.json` is the fixed 500-question subset selected from
HotpotQA's `hotpot_dev_distractor_v1.json`, retaining examples with ten candidate
paragraphs. The original question, answer, context, and supporting-fact fields
are preserved.

Source: [HotpotQA](https://hotpotqa.github.io/). The dataset is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The modification is selection of the evaluation subset.

Dataset authors and citation:

> Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
> Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A Dataset for
> Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.

HotpotQA material reproduced in `results/raw/` and `results/judged/` retains
its dataset license. The repository's MIT license does not relicense these
data. Preserve this attribution when redistributing them.

Model outputs and judge annotations remain subject to applicable model and
service terms. Model weights and third-party software are installed separately
and retain their upstream licenses.
