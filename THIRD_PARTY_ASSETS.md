# Data attribution

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
