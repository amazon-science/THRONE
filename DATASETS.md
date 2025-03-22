# Prepare Datasets for THRONE
Our work evaluates a given large vision-language model (LVLM) using two datasets:
- [Microsoft COCO](https://cocodataset.org/#home)
- [Objects 365](https://www.objects365.org/overview.html)

## Data Directory Structure
```
./$ROOT/data
├── coco
│   ├── annotations
│   │   └── instances_val2017.json  # from 2017 Train/Val annotations
│   └── val2017  # folder containing 2017 Val images
└── objects365
    ├── images
    │   ├── v1
    │   │   └── patch{0,...,15}   # folders containing val images
    │   └── v2
    │       └── patch{16,...,43}  # folders containing val images
    ├── zhiyuan_objv2_val_fixname.json
    └── zhiyuan_objv2_val.json
```
The COCO data can pretty much be downloaded as is.
However the process for Objects365 is slightly more involved.
After downloading the data listed above, follow the instructions
[here](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md#objects365) from the Detic work to form
`zhiyuan_objv2_val_fixname.json` which is the corrected label names THRONE requires.
