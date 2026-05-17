# AutoKaggle

AutoKaggle is a multi-agent framework for tabular Kaggle-style competitions. It coordinates a Reader, Planner, Developer, Reviewer, and Summarizer across a fixed six-phase workflow, then writes the intermediate plans, code, feedback, and outputs back into the competition workspace.

Project home: https://m-a-p.ai/AutoKaggle.github.io/

Paper: https://arxiv.org/abs/2410.20424.pdf

License: [LICENSE.md](LICENSE.md)

![AutoKaggle overview](./mdPICs/kaggle_main.png)

## What Runs Where

`framework.py` is the main entry point. It creates a `State`, hands it to the `SOP`, and keeps stepping until the competition finishes. The `SOP` instantiates the phase agents from `multi_agents/agents/`, selects models from `multi_agents/config.json`, and repeats a phase when the reviewer score is below the completion threshold.

The six phases are:

1. Understand Background
2. Preliminary Exploratory Data Analysis
3. Data Cleaning
4. In-depth Exploratory Data Analysis
5. Feature Engineering
6. Model Building, Validation, and Prediction

The cleaner, feature engineering, and final modeling phases also run phase-specific checks against the generated CSVs and submission file before the workflow advances.

## Repository Map

- `framework.py` - command-line entry point for a single competition run.
- `run_multi_agents.sh` - batch runner for the preset competition list.
- `model_selector.py` - interactive helper for changing agent model profiles in `multi_agents/config.json`.
- `api_handler.py` - legacy OpenAI client wrapper used by older helper paths.
- `Architecture.md` - deeper architecture notes.
- `multi_agents/` - orchestration package with agents, prompts, providers, memory, and tools.
- `multi_agents/README.md` - how to add custom ML tools.
- `multi_agents/tools/ml_tools_doc/` - markdown docs used for tool retrieval.
- `mdPICs/` - screenshots used in this README.

## Setup

1. Create or activate a Python 3.11 environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure API credentials.

- Preferred: copy `.env.template` to `.env` and set `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and optional `ANTHROPIC_API_KEY`.
- Backward-compatible fallback: keep `api_key.txt` at the project root with the same credentials if you want to use the legacy path.
- Keep an OpenAI key configured if you want the tool-retrieval path to work, even when the agents themselves use Anthropic models.

The provider factory loads `.env` automatically, and model selection is inferred from the model name you pass to `framework.py`.

4. Prepare competition data in `multi_agents/competition/<competition>/`.

```text
competition/
├── train.csv
├── test.csv
├── sample_submission.csv
└── overview.txt
```

`overview.txt` should contain the Kaggle "Overview" and "Data" sections so the Reader can summarize the task.

## Run

Single competition run:

```bash
python framework.py --competition titanic --model gpt-4o
```

Anthropic example:

```bash
python framework.py --competition titanic --model claude-sonnet-4-6
```

Batch runs:

```bash
bash run_multi_agents.sh
```

On Windows, run the bash script from Git Bash or WSL.

Optional model profile helper:

```bash
python model_selector.py
```

This updates the agent-to-model mapping in `multi_agents/config.json`.

## Output Layout

Phase artifacts are written under `multi_agents/competition/<competition>/<phase_dir>/`, where the phase directories are:

- `understand_background`
- `pre_eda`
- `data_cleaning`
- `deep_eda`
- `feature_engineering`
- `model_build_predict`

Typical files include `competition_info.txt`, `markdown_plan.txt`, `json_plan.json`, `*_code.py`, `*_run_code.py`, `single_phase_code.txt`, `review.json`, `report.txt`, `memory.json`, and phase-specific CSV outputs such as `cleaned_train.csv`, `cleaned_test.csv`, `processed_train.csv`, `processed_test.csv`, and `submission.csv`.

Batch runs copy completed outputs into `multi_agents/experiments_history/<competition>/<model>/<dest_dir_param>/<run_number>/`.

## Tool Library

`multi_agents/tools/ml_tools.py` contains reusable functions for:

- missing-value handling and column dropping
- outlier detection and cleanup
- duplicate removal and type conversion
- datetime formatting
- categorical encoding
- feature selection and scaling
- PCA, RFE, polynomial features, and feature combinations
- model selection and validation

The corresponding markdown docs live in `multi_agents/tools/ml_tools_doc/`, and the Planner uses them to retrieve tool descriptions. If you add a new tool, update the implementation, `function_to_schema.json`, and the matching markdown docs.

## Reported Results

The paper reports:

- 85% validation submission rate
- 0.82 comprehensive score

![Main results](./mdPICs/main_results.png)
![Average NPS](./mdPICs/average_nps.png)

## Citation

```text
@misc{li2024autokagglemultiagentframeworkautonomous,
      title={AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions},
      author={Ziming Li and Qianbo Zang and David Ma and Jiawei Guo and Tianyu Zheng and Minghao liu and Xinyao Niu and Yue Wang and Jian Yang and Jiaheng Liu and Wanjun Zhong and Wangchunshu Zhou and Wenhao Huang and Ge Zhang},
      year={2024},
      eprint={2410.20424},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.20424},
}
```

## Disclaimer

This project, "AutoKaggle," is not affiliated with, endorsed by, or officially associated with Kaggle or Google in any way. The use of the name "Kaggle" is solely to indicate compatibility with Kaggle competitions. All trademarks, logos, and brand names are the property of their respective owners. We respect Kaggle's brand guidelines and are in the process of rebranding to better reflect our independence. For further details or concerns, please contact us.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
