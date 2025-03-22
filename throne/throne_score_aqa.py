# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import glob
import json
from collections import Counter

import pycocotools.coco as coco
import torch


def process_aqa_instance_preds(aqa_responses, num_categories, sorted_categories_idx_to_name):
    """processes the evaluator model predictions for an instance given different classes and prompts into a nested list of present categories"""
    responses_by_prompt = [
        aqa_responses[i * num_categories : (i + 1) * num_categories]
        for i in range(len(aqa_responses) // num_categories)
    ]
    present_classes_by_prompt = [
        sorted(
            [
                sorted_categories_idx_to_name[i]
                for i, a in enumerate(single_prompt_responses)
                if a.lower().strip() == "yes"
            ]
        )
        for single_prompt_responses in responses_by_prompt
    ]
    return present_classes_by_prompt


def prepare_aqa_preds(eval_model_preds_paths, sorted_categories_idx_to_name):
    """aggregates aqa json responses for different prompts and different evaluator models.
    output is a dictionary mapping each image_id to a nested list of present categories
    each list inside the nested list is of the category names present given the eval model and prompt.
    For example with 2 prompts and 3 models the nested list would be doubly nested with 3 outer lists each containing 2 inner lists of categories present

    eval_model_preds_paths paths of saved aqa output json files
    sorted_categories_idx_to_name mapping the sorted idx of a class to it's name
    returns Dict[int, List[List[List[str]]]] # image_id -> eval_model list -> prompt list -> present classes
    """
    preds_by_eval_model_list = []
    for eval_model_preds_path in eval_model_preds_paths:
        with open(eval_model_preds_path, "r") as f:
            raw_preds = json.load(f)
            preds_by_eval_model = {
                int(k): process_aqa_instance_preds(
                    v,
                    num_categories=len(sorted_categories_idx_to_name),
                    sorted_categories_idx_to_name=sorted_categories_idx_to_name,
                )
                for (k, v) in raw_preds.items()
            }
            preds_by_eval_model_list.append(preds_by_eval_model)
    combined_preds = dict()
    for k in preds_by_eval_model_list[0].keys():
        combined_preds[k] = [
            preds_by_eval_model[k] for preds_by_eval_model in preds_by_eval_model_list
        ]
    return combined_preds


def majority_vote_collapsing(aqa_present_classes, voting_threshold):
    """aggregate predictions for different prompts and eval models requiring at least voting_threshold votes for an object being present"""
    collapsed_aqa_present_classes = dict()
    collapsed_aqa_disqualified_classes = dict()
    for image_id, nested_present_classes in aqa_present_classes.items():
        instance_counter = Counter()
        present_categories = [present_category for inner_nested in nested_present_classes for present_category in inner_nested]  # count present categories
        for present_category in present_categories:
            instance_counter.update(present_category) 
        num_votes = len(present_categories)
        final_instance_classes = [
            present_category
            for (present_category, counts) in instance_counter.items()
            if counts >= voting_threshold
        ]
        final_disqualified_classes = [
            present_category
            for (present_category, counts) in instance_counter.items()
            if num_votes - voting_threshold < counts < voting_threshold
        ]
        collapsed_aqa_present_classes[image_id] = final_instance_classes
        collapsed_aqa_disqualified_classes[image_id] = final_disqualified_classes
    return collapsed_aqa_present_classes, collapsed_aqa_disqualified_classes


def micro_avergaing_metrics(predicted, disqualified, gt):
    # micro-averaging strategy used (e.g. computing P, R at the instance level
    print("micro-averaging (at image level)")
    precision = []
    recall = []
    for k in predicted.keys():
        disqualified_k = set(disqualified[k])
        gt_present = [x for x in gt[k] if x not in disqualified_k]
        pred_present = [x for x in predicted[k] if x not in disqualified_k]
        if len(set(pred_present)) > 0 and len(set(gt_present)) > 0:
            p = len(set(gt_present).intersection(set(pred_present))) / len(set(pred_present))
            r = len(set(gt_present).intersection(set(pred_present))) / len(set(gt_present))
        else:
            # print('encountered 0')
            p = 0
            r = 0
        precision.append(p)
        recall.append(r)

    precision = torch.Tensor(precision)
    recall = torch.Tensor(recall)
    f_one = 2 * precision * recall / (precision + recall)
    num_nans = torch.nonzero(torch.isnan(f_one)).numel()
    f_one = f_one.nanmean()
    f_one_half = (
        1.25 * precision * recall / (0.25 * precision + recall)
    ).nanmean()  # https://en.wikipedia.org/wiki/F-score#F%CE%B2_score
    print(
        f"F1: {f_one:.3f}\nF0.5: {f_one_half:.3f}\nP: {precision.nanmean():.3f}\nR:{recall.nanmean():.3f}\nN:{len(precision)}\nNaNs: {num_nans}"
    )


def classwise_averaging_metrics(predicted, disqualified, gt, sorted_categories_idx_to_name):
    name_to_idx = {v: k for (k, v) in sorted_categories_idx_to_name.items()}
    num_categories = len(sorted_categories_idx_to_name)
    class_predicted = [0] * num_categories
    class_correct = [0] * num_categories
    class_gt = [0] * num_categories
    for k in predicted.keys():
        disqualified_k = set(disqualified[k])
        for gt_category in gt[k]:
            if gt_category not in disqualified_k:
                class_gt[name_to_idx[gt_category]] += 1
        for predicted_category in predicted[k]:
            if predicted_category not in disqualified_k:
                class_predicted[name_to_idx[predicted_category]] += 1
                if predicted_category in gt[k]:
                    class_correct[name_to_idx[predicted_category]] += 1
    class_predicted = torch.Tensor(class_predicted)
    class_correct = torch.Tensor(class_correct)
    class_gt = torch.Tensor(class_gt)
    precision = class_correct / class_predicted
    recall = class_correct / class_gt
    f_one = 2 * precision * recall / (precision + recall)
    f_one_half = 1.25 * precision * recall / (0.25 * precision + recall)
    print(
        f"F1 (classwise): {f_one.nanmean():.3f}\nF0.5 (classwise): {f_one_half.nanmean():.3f}\nP (classwise): {precision.nanmean():.3f}\nR (classwise):{recall.nanmean():.3f}"
    )


def throne_metrics_from_eval(args):
    eval_model_preds_paths = glob.glob(args.model_eval_path + "/*.json")
    if type(args.thresholds) == int:
        thresholds = [args.thresholds]
    else:
        thresholds = list(args.thresholds)
    coco_dataset = coco.COCO(args.coco_val_ann_path)
    sorted_categories = sorted(coco_dataset.cats.values(), key=lambda x: x["id"])
    sorted_categories_idx_to_name = {
        i: cat["name"] for i, cat in enumerate(sorted_categories)
    }  # note coco_id and idx are different
    categories_coco_id_to_name = {cat["id"]: cat["name"] for cat in sorted_categories}
    # construct aqa present classes for each image
    aqa_present_classes = prepare_aqa_preds(eval_model_preds_paths, sorted_categories_idx_to_name)
    # construct GT present classes for each image
    gt_present_classes = dict()
    for image_id in aqa_present_classes.keys():
        ann_ids = coco_dataset.getAnnIds(imgIds=[image_id])
        anns = coco_dataset.loadAnns(ann_ids)
        gt_instance_present_classes = sorted(
            list(set([categories_coco_id_to_name[ann["category_id"]] for ann in anns]))
        )
        gt_present_classes[image_id] = gt_instance_present_classes

    for threshold in args.thresholds:
        print(f"voting threshold: {threshold}")
        collapsed_aqa_present_classes, collapsed_aqa_disqualified_classes = majority_vote_collapsing(
            aqa_present_classes, voting_threshold=threshold
        )
        if "cls" in args.metric_strategy or "class" in args.metric_strategy:
            classwise_averaging_metrics(
                collapsed_aqa_present_classes, collapsed_aqa_disqualified_classes, gt_present_classes, sorted_categories_idx_to_name
            )

        elif "micro" in args.metric_strategy:
            micro_avergaing_metrics(collapsed_aqa_present_classes, collapsed_aqa_disqualified_classes, gt_present_classes)
        else:
            raise NotImplementedError(f"{args.metric_strategy} not implemented!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_val_ann_path", type=str, default=None)
    parser.add_argument(
        "--model_eval_path",
        type=str,
        default=None,
        help="path to directory with grouped aqa evaluated models",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[9],
        help="voting thresholds, 9 is the unanimous voting with 3 models, 3 prompts",
    )
    parser.add_argument(
        "--metric_strategy", type=str, default=None, help="between classwise and micro"
    )
    args = parser.parse_args()
    throne_metrics_from_eval(args)
