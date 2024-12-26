# Open-Ended LLM Judgement

- We currently support four benchmarks
  - MT-Bench
  - Vicuna benchmark
  - Advbench
  - GPT4-Refer

You could firstly run `utils/merge_lora.py` to merge LORA to the base model.

We provide all scripts in `scripts` for convenience. 

**About GPT4-Refer**

We select instructions from the test set of the dataset and use the model under test to generate responses. Then, we use GPT-4 to score these responses (1-10).
For more information about MT-Bench, Vicuna Bench and AdvBench, see [README.md](https://github.com/rui-ye/OpenFedLLM/blob/main/evaluation/open_ended/README.md)

## Step 1: Generate answers

Use `bash gen_answer_xx.sh` to generate answers.
`gen_answer_bench.sh` is used for GPT4-Refer.

**TODO**
- `gpus`
- `lora_pathes` (the lora path of your model)
- `bench_name` (for `gen_answer_bench.sh`)

## Step 2: Generate judgments
Use `bash gen_judge_xx.sh` to generate judgments. `gen_judge_bench` is used for GPT4-Refer.

**TODO**

- OPENAI_API_KEY
- `model_answer_list` (answers generate by the model)

## Step 3: Show results
Use `bash show_results_xx.sh` to show results.

For AdvBench, the results will be directly output after generating answers.



## Citation

For MT-Bench and Vicuna Benchmark:
```

```

For Advbench:
```

```

