# THRONE
This code implements a benchmark for evaluating object hallucination in free-form response outputs of VLM models.\
THRONE aims to be a comprehensive and flexible benchmark for un-guided free-response generations. \
This is based on Prannay Kaul's intern project published at [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kaul_THRONE_An_Object-based_Hallucination_Benchmark_for_the_Free-form_Generations_of_CVPR_2024_paper.html) \
with the Title of ***THRONE: An Object-based Hallucination Benchmark for the Free-form Generations of Large Vision-Language Models***

## Installation
See [INSTALL.md](./INSTALL.md)

## Datasets
See [DATASETS.md](./DATASETS.md)

## Running
Throne evaluation has 3 distinct steps, in the first step a VLM model is prompted to generate free-response answers given an image.
In the second step the generations are evaluated using a set of evaluator LLMs. Lastly in the 3rd step, we score the Throne metrics the LLM interpretations of the VLM.

The full flow of THRONE can be run from a single file `throne/scripts/one-click.sh`,
e.g. to run LLaVA-v1.6-Mistral-7b on COCO:
```bash
cd $ROOT/throne
conda activate throne
PRETRAINED_MODELS=/path/to/pretrained/models/directory MODEL='LLaVAMistral' DATASET='coco_subset' bash scripts/one-click.sh
```
The three steps are clearly marked in `scripts/one-click.sh` making use of `throne/throne_generate.py`,
`throne/throne_aqa_evaluation.py` and `throne/throne_score_aqa.py`, respectively.
The script has been tested on a g5.48xlarge EC2 instance, but a smaller machine might also work.

To evaluate a new model, extend the `Evaluatee` class in `throne/evaluated_models.py`, and implement the placeholder functions, using the classes for supported models as examples.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This repository is licensed under the CC-BY-NC-4.0 License.

