# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import contextlib
import io
import json
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from throne_constants import Q_LIST, I_LIST

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def simple_collate(batch):
    total = len(batch[0])
    return [[b[i] for b in batch] for i in range(total)]


def get_question_lists(cat_list_, singular_q="", plural_q=""):
    if isinstance(cat_list_, dict):
        cat_list = sorted(cat_list_.values(), key=lambda x: x["id"])
    else:
        cat_list = cat_list_

    questions_list_yes_no = []
    for cat in cat_list:
        if cat["name"] in {"skis", "scissors"}:
            questions_list_yes_no.append(plural_q.format(cat["name"]))
        else:
            name = "necktie" if cat["name"] == "tie" else cat["name"]
            article = "an" if name[0] in "aeiou" else "a"
            questions_list_yes_no.append(singular_q.format(article, name))
    return questions_list_yes_no


def init_distributed_mode():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def run_extractive_qa(args):
    save_dir, save_file = os.path.split(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    save_file, ext = os.path.splitext(save_file)
    device_map = {"": LOCAL_RANK}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.coco_file)
    with open(args.response_file, "r") as f:
        responses_list = json.load(f)["responses"]
    responses = {int(a[1]): a[2] for a in responses_list}

    qa_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.evaluator_model_path, device_map=device_map, torch_dtype=torch.bfloat16
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model_path)
    coco_q_yn = []
    for qa_tuple in Q_LIST[: args.M]:  # number of question formats
        coco_q_yn += get_question_lists(coco_gt.cats, singular_q=qa_tuple[0], plural_q=qa_tuple[1])
    print(len(coco_q_yn))

    prompt_format = (
        "Text: {context}\n\n" + I_LIST[0] + "\n\n{question}"
    )  # 1 instruction as in paper
    iids = sorted(coco_gt.getImgIds())
    response_iids = sorted(responses.keys())
    iids = sorted(set(response_iids).intersection(set(iids)))
    iids_local = iids[LOCAL_RANK::WORLD_SIZE]
    answers_worker = simplified_aqa_coco_ids(
        qa_model, qa_tokenizer, prompt_format, coco_q_yn, responses, iids_local, args
    )

    if WORLD_SIZE > 1:
        init_distributed_mode()
        file_name = f"{save_file}_{args.evaluator_model_path.replace('/', '_')}_{LOCAL_RANK}_{WORLD_SIZE}.json"
    else:
        file_name = f"{save_file}_{args.evaluator_model_path.replace('/', '_')}.json"
        all_answers = answers_worker
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(answers_worker, f)

    # group answers_worker
    if WORLD_SIZE > 1:
        dist.barrier()
        if LOCAL_RANK == 0:
            print("grouping workers jsons")
            all_answers = dict()
            for worker in range(WORLD_SIZE):
                with open(
                    os.path.join(
                        save_dir,
                        f"{save_file}_{args.evaluator_model_path.replace('/', '_')}_{worker}_{WORLD_SIZE}.json",
                    ),
                    "r",
                ) as f:
                    outputs = json.load(f)
                    all_answers.update(outputs)
            os.makedirs(os.path.join(save_dir, "combined"), exist_ok=True)
            with open(
                os.path.join(
                    save_dir,
                    "combined",
                    f"{save_file}_{args.evaluator_model_path.replace('/', '_')}.json",
                ),
                "w",
            ) as f:
                json.dump(all_answers, f)


class AQADataset(Dataset):
    def __init__(self, responses, coco_ids, questions, prompt, response_truncation=1024):
        self.responses = responses
        self.coco_ids = coco_ids
        self.questions = questions
        self.prompt = prompt
        self.response_truncation = response_truncation

    def __len__(self):
        return (len(self.coco_ids)) * len(self.questions)

    @property
    def num_questions(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_idx = idx % len(self.questions)
        response_idx = idx // len(self.questions)
        coco_id = self.coco_ids[response_idx]
        return (
            self.prompt.format(
                context=self.responses[coco_id][: self.response_truncation],
                question=self.questions[question_idx],
            ),
            coco_id,
            question_idx,
        )


@torch.inference_mode()
def simplified_aqa_coco_ids(model, tokenizer, prompt, questions, context, coco_ids, args):
    dataset = AQADataset(context, coco_ids, questions, prompt)
    all_answers = defaultdict(lambda: [None] * dataset.num_questions)
    dataloader = DataLoader(
        dataset, batch_size=args.per_device_batch_size, shuffle=False, num_workers=2
    )
    device = model.device
    for aqa_prompts, coco_idx, question_idx in tqdm(dataloader):
        input_ids = tokenizer(aqa_prompts, padding=True, return_tensors="pt").input_ids.to(device)
        # print(input_ids.shape)
        output_ids = model.generate(
            input_ids,
            use_cache=True,
            max_length=16,
            do_sample=False,
        )
        output_strs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for o, q, coco in zip(output_strs, question_idx, coco_idx):
            all_answers[int(coco)][int(q)] = o
    return all_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, default=None)
    parser.add_argument("--coco_file", type=str, default=None)
    parser.add_argument("--evaluator_model_path", type=str, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--M",
        type=int,
        help="M is the number of question formats as defined in the paper",
        default=3,
    )
    args = parser.parse_args()
    run_extractive_qa(args)
