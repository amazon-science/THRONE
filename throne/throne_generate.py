# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
from io import BytesIO

from transformers import set_seed

import requests
import torch.distributed as dist
from torch.utils.data import Sampler
import tqdm
from PIL import Image
from evaluated_models import get_evaluated_model
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


GENERATION_PROMPT = "Describe this image in detail."

class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """
    
    def __init__(self, size: int, rank: int, world_size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = rank
        self._world_size = world_size
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)


    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)



def init_distributed_mode():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


class COCOImageDataset(Dataset):
    def __init__(self, coco_gt: COCO, coco_image_dir, coco_ids, transform=None, skip_load_image=False):
        self.coco_gt = coco_gt
        self.image_dir = coco_image_dir
        self.coco_ids = sorted(coco_ids)
        self.transform = transform
        self.skip_load_image = skip_load_image

    def __len__(self):
        return len(self.coco_ids)

    def __getitem__(self, index):
        coco_id = self.coco_ids[index]
        image_file = self.coco_gt.loadImgs(coco_id)[0]["file_name"]
        if not self.skip_load_image:
            image = load_image(os.path.join(self.image_dir, image_file))
            if self.transform:
                image = self.transform(image)["pixel_values"][0]
        else:
            image = os.path.join(self.image_dir, image_file)

        return coco_id, image


def simple_collate(batch):
    total = len(batch[0])
    return [[b[i] for b in batch] for i in range(total)]


def batch_generate_with_llava(
    model, generation_prompt, image_dataloader, rank=0
):
    response_tuple = []

    gen_prompt_idx = 0
    input_ids = model.tokenize_prompt(generation_prompt)
    if rank == 0:
        image_dataloader = tqdm.tqdm(image_dataloader)

    for idx, (indices, imgs) in enumerate(image_dataloader):
        batch_outputs = model.generate_batch(input_ids, imgs)
        
        response_tuple += list(zip([gen_prompt_idx] * len(indices), indices, batch_outputs))
    return response_tuple


def generate_responses_with_any(args):
    save_dir, save_file = os.path.split(args.save_path)
    save_file, ext = os.path.splitext(save_file)

    model = get_evaluated_model(args.modelclass, args)
    generation_prompt = model.format_prompt(GENERATION_PROMPT)

    model.load()

    coco_gt = COCO(args.coco_file)
    if not args.coco_subset:
        coco_ids = sorted(coco_gt.getImgIds())
    else:
        with open(args.coco_subset) as f:
            coco_ids = sorted(int(x.strip()) for x in f.readlines())
    dataset = COCOImageDataset(
        coco_gt=coco_gt, coco_image_dir=args.coco_image_dir, coco_ids=coco_ids, transform=None,
        skip_load_image=args.skip_load_img
    )
    ddp = "RANK" in os.environ
    if not ddp:
        image_loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=simple_collate,
            drop_last=False,
        )
        outputs = batch_generate_with_llava(
            model,
            generation_prompt,
            image_dataloader=image_loader,
        )
        results = dict(prompts=[GENERATION_PROMPT], responses=outputs)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_file}.json"), "w") as f:
            json.dump(results, f)
    else:
        init_distributed_mode()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.barrier()

        sampler = InferenceSampler(len(dataset), world_size=world_size, rank=rank)
        image_loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=simple_collate,
            sampler=sampler,
            drop_last=False,
        )
        outputs = batch_generate_with_llava(
            model,
            generation_prompt,
            image_dataloader=image_loader,
            rank=rank,
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_file}_{rank}_{world_size}.json"), "w") as f:
            json.dump(outputs, f)
        dist.barrier()
        if rank == 0:
            print("grouping workers jsons")
            combined_outputs = []
            for worker in range(world_size):
                with open(
                    os.path.join(save_dir, f"{save_file}_{worker}_{world_size}.json"), "r"
                ) as f:
                    outputs = json.load(f)
                    combined_outputs += outputs
            results = dict(prompts=[GENERATION_PROMPT], responses=combined_outputs)
            with open(os.path.join(save_dir, f"{save_file}.json"), "w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_file", type=str, default=None)
    parser.add_argument("--coco_image_dir", type=str, default=None)
    parser.add_argument("--coco_subset", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--skip_load_img", action="store_true")
    subparsers = parser.add_subparsers(help='Model-specific options')

    parser_llava = subparsers.add_parser('LLaVA', help='Specific options for LLaVA')
    parser_llava.set_defaults(modelclass='LLaVA')
    parser_llava.add_argument("--model_path", type=str, default=None)
    parser_llava.add_argument("--model_base", type=str, default=None)
    parser_llava.add_argument("--conv_template_name", type=str, default=None)

    parser_minigpt = subparsers.add_parser('MiniGPT', help='Specific options for MiniGPT')
    parser_minigpt.set_defaults(modelclass='MiniGPT')
    parser_minigpt.add_argument("--cfg_path", type=str, default="src/mini-gpt/eval_configs/minigpt4_eval.yaml")
    parser_minigpt.add_argument("--model_path", type=str, default="")

    parser_minigpt2 = subparsers.add_parser('MiniGPTv2', help='Specific options for MiniGPTv2')
    parser_minigpt2.set_defaults(modelclass='MiniGPTv2')
    parser_minigpt2.add_argument("--cfg_path", type=str, default="src/mini-gpt/eval_configs/minigptv2_eval.yaml")
    parser_minigpt2.add_argument("--model_path", type=str, default="")

    parser_instructblip = subparsers.add_parser('InstructBLIP', help='Specific options for InstructBlip')
    parser_instructblip.set_defaults(modelclass='InstructBLIP')
    parser_instructblip.add_argument("--model_path", type=str, default=None)

    parser_mplugowl = subparsers.add_parser('mPLUGOwl', help='Specific options for mPLUGOwl')
    parser_mplugowl.set_defaults(modelclass='mPLUGOwl')
    parser_mplugowl.add_argument("--model_path", type=str, default=None)

    parser_lrvinstruct = subparsers.add_parser('LRVInstruct', help='Specific options for LRVInstruct')
    parser_lrvinstruct.set_defaults(modelclass='LRVInstruct')
    parser_lrvinstruct.add_argument("--model_base", type=str, default=None)
    parser_lrvinstruct.add_argument("--model_path", type=str, default=None)

    parser_adapter = subparsers.add_parser('LLaMAAdapter', help='Specific options for LLaMAAdapter')
    parser_adapter.set_defaults(modelclass='LLaMAAdapter')
    parser_adapter.add_argument("--model_name", type=str, default=None)
    parser_adapter.add_argument("--llama_model_path", type=str, default=None)

    parser_otter = subparsers.add_parser('Otter', help='Specific options for Otter')
    parser_otter.set_defaults(modelclass='Otter')
    parser_otter.add_argument("--model_path", type=str, default=None)

    args = parser.parse_args()
    generate_responses_with_any(args)
