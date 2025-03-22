# Installation Instructions for THRONE

## Models and Third-Party Code

Model families and specific models which this code supports:
- LLaVA family [code](https://github.com/haotian-liu/LLaVA/tree/c121f0432da27facab705978f83c4ada465e46fd)
    - LLaVA-v1.6-Mistral-7b [checkpoint](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)
    - LLaVA-v1.5-Vicuna-7b [checkpoint](https://huggingface.co/liuhaotian/llava-v1.5-7b)
    - LLaVA-v1.3-Vicuna-7b [checkpoint](https://huggingface.co/MaoXun/llava-lora-vicuna-7b-v1.3)
- MiniGPT family [code](https://github.com/Vision-CAIR/MiniGPT-4/tree/d94738a7626ec43eba6c2cddf3cd2043f1a9689a)
    - MiniGPT-4-Vicuna-7b [checkpoint](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)
    - MiniGPT-v2-Llama2-chat-7b [checkpoint](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing)
- mPLUG-Owl family [code](https://github.com/X-PLUG/mPLUG-Owl/tree/e621a93ccfdd8480f2ed24c7d96659c13d14d11b)
    - mPLUG-Owl-Llama-7b [checkpoint](https://huggingface.co/MAGAer13/mplug-owl-llama-7b)
    - LRV-Instruction-v2 [checkpoint](https://github.com/FuxiaoLiu/LRV-Instruction/blob/main/download.txt#L24)
- LlamaAdapter family [code](https://github.com/OpenGVLab/LLaMA-Adapter/tree/32f5118c291d02cb2c6143c1caf1190f7bc44625)
    - LlamaAdapter-V2 [checkpoint](https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth)
    - LlamaAdapter-V2.1 [checkpoint](https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.1.0/d26d107eec32127ac86ef1997cf7169de1c56a59c539fc1258c6798b969e289c_LORA-BIAS-7B-v21.pth)
- Otter family [code](https://github.com/Luodian/Otter/tree/1e7eb9a6fb12ef410082e796c463b99495637b85)
    - Otter-Image-MPT-7b [checkpoint](https://huggingface.co/luodian/OTTER-Image-MPT7B)
- InstructBLIP family
    - InstructBLIP-Vicuna-7b [checkpoint](https://huggingface.co/Salesforce/instructblip-vicuna-7b)

Many of the above checkpoints require other language models to be downloaded such as Llama2 or Vicuna.
We refer the reader to the above linked code bases to determine which other checkpoints may be required.
Note the above code links also point to _specific commits_ which may be installed using submodules.

## Example Setup with Conda
```bash
conda create --name throne python=3.10 -y
conda activate throne
git clone https://github.com/amazon-science/THRONE.git --recurse-submodules
cd THRONE
pip install -r requirements.txt
```
If you forget to add `--recurse-submodules`, do `git submodule init` and then `git submodule update`.

## Directory structure
```
./$ROOT
├── data
│   ├── coco
│   └── objects365
├── third_party
│   ├── LLaMA-Adapter
│   ├── LLaVA
│   ├── MiniGPT-4
│   ├── mPLUG-Owl
│   └── Otter
└── throne
```
Reminder that any third-party code must be downloaded matching the structure above _and_ the correct commit must be checked out.

See [DATASETS.md](DATASETS.md) for more information on the datasets used and the directory structure required.


