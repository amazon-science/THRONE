# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import torch


class Evaluatee(object):
    def __init__(self, args):
        self.args = args

    def load(self):
        raise NotImplementedError()

    def format_prompt(self, prompt):
        return prompt

    def tokenize_prompt(self, formatted_prompt):
        raise NotImplementedError()

    def generate_batch(self, input_ids, imgs):
        raise NotImplementedError()


class LLaVAMistralVicuna(Evaluatee):
    def __init__(self, args):
        super().__init__(args)
        import llava.constants
        import llava.conversation
        import llava.mm_utils
        import llava.model.builder
        self.llava = llava

    def load(self):
        model_path = self.args.model_path
        model_base = self.args.model_base
        model_name = self.llava.mm_utils.get_model_name_from_path(model_path)
        self.tokenizer, self.model, image_processor, self.context_len = self.llava.model.builder.load_pretrained_model(
            model_path, model_base, model_name, device="cuda"
        )
        config = self.model.config
        config.image_aspect_ratio = "pad"
        self.image_list_processor = lambda images: self.llava.mm_utils.process_images(images, image_processor, config)

    def format_prompt(self, prompt):
        qs = prompt
        qs = self.llava.constants.DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = self.llava.conversation.conv_templates[self.args.conv_template_name].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def tokenize_prompt(self, formatted_prompt):
        input_ids = self.llava.mm_utils.tokenizer_image_token(
            formatted_prompt, self.tokenizer, self.llava.constants.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).cuda()
        return input_ids

    def generate_batch(self, input_ids, imgs):
        if self.image_list_processor is not None:
            img_tensor = self.image_list_processor(imgs)
        else:
            img_tensor = imgs
        input_ids_batch = input_ids.unsqueeze(0).repeat(
            ((len(img_tensor),) + (1,) * len(input_ids.shape))
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_batch,
                images=img_tensor.half().cuda(),
                image_sizes=[im.size for im in imgs],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                max_new_tokens=1024,
                use_cache=True,
            )
        batch_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_outputs = [o.strip() for o in batch_outputs]
        return batch_outputs


class InstructBLIP(Evaluatee):
    def __init__(self, args):
        super().__init__(args)
        import transformers.models.instructblip.modeling_instructblip
        import transformers.models.instructblip.processing_instructblip
        import transformers.models.blip_2.modeling_blip_2
        import transformers.models.blip_2.processing_blip_2
        self.transformers = transformers
    
    def load(self):
        self.processor = self.transformers.models.instructblip.processing_instructblip.InstructBlipProcessor.from_pretrained(
            self.args.model_path, use_fast=False
        )
        self.model = self.transformers.models.instructblip.modeling_instructblip.InstructBlipForConditionalGeneration.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        ).cuda()
    
    def format_prompt(self, prompt):
        self.qs = prompt
        return None

    def tokenize_prompt(self, formatted_prompt):
        if formatted_prompt is not None:
            raise ValueError("formatted_prompt must be None for InstructBLIP/BLIP2")
        return None

    def generate_batch(self, input_ids, imgs):
        inputs = self.processor(
            images=imgs,
            text=[self.qs for _ in range(len(imgs))],
            return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                # generation_config=gen_config,
                # images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                # stopping_criteria=[stopping_criteria],
        )
        # input_ids = inputs.input_ids
        # input_token_len = input_ids.shape[1]

        batch_outputs = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        batch_outputs = [o.strip() for o in batch_outputs]
        return batch_outputs


class BLIP2(InstructBLIP):
    def __init__(self, args):
        super().__init__(args)
    
    def load(self):
        self.processor = self.transformers.models.blip_2.processing_blip_2.Blip2Processor.from_pretrained(
            self.args.model_path
        )
        self.model = self.transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGeneration.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        ).cuda()
    
    def format_prompt(self, prompt):
        self.qs = "Question: {} Answer:".format(prompt)
        return None
        

class mPLUGOwl(Evaluatee):
    def __init__(self, args):
        super().__init__(args)
        import mplug_owl.modeling_mplug_owl
        import mplug_owl.tokenization_mplug_owl
        import mplug_owl.processing_mplug_owl
        self.mplug_owl = mplug_owl
    
    def load(self):
        self.model = self.mplug_owl.modeling_mplug_owl.MplugOwlForConditionalGeneration.from_pretrained(
            self.args.model_path, torch_dtype=torch.float16
        ).cuda()
        image_processor = self.mplug_owl.processing_mplug_owl.MplugOwlImageProcessor.from_pretrained(
            self.args.model_path
        )
        tokenizer = self.mplug_owl.tokenization_mplug_owl.MplugOwlTokenizer.from_pretrained(
            self.args.model_path
        )
        self.processor = self.mplug_owl.processing_mplug_owl.MplugOwlProcessor(
            image_processor, tokenizer
        )

    def format_prompt(self, prompt):
        self.qs = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: {}
AI: '''.format(prompt)]
        return None

    def tokenize_prompt(self, formatted_prompt):
        if formatted_prompt is not None:
            raise ValueError("formatted_prompt must be None for mPLUGOwl")
        return None

    def generate_batch(self, input_ids, imgs):
        inputs = self.processor(
            images=imgs,
            text=self.qs,
            return_tensors='pt'
        ).to(self.model.device)
        inputs["input_ids"] = inputs["input_ids"].expand(len(imgs), -1)
        inputs["attention_mask"] = inputs["attention_mask"].expand(len(imgs), -1)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                # stopping_criteria=[stopping_criteria],
        )
        
        batch_outputs = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_outputs = [o.strip() for o in batch_outputs]
        return batch_outputs
        

class LRVInstruct(Evaluatee):
    def __init__(self, args):
        super().__init__(args)
        import mplug_owl.modeling_mplug_owl
        import mplug_owl.tokenization_mplug_owl
        import mplug_owl.processing_mplug_owl
        self.mplug_owl = mplug_owl
        self.model_path = args.model_path
    
    def load(self):
        from peft import LoraConfig, get_peft_model
        model_base = self.mplug_owl.modeling_mplug_owl.MplugOwlForConditionalGeneration.from_pretrained(
            self.args.model_base, torch_dtype=torch.float16
        ).cuda()
        image_processor = self.mplug_owl.processing_mplug_owl.MplugOwlImageProcessor.from_pretrained(
            self.args.model_base
        )
        tokenizer = self.mplug_owl.tokenization_mplug_owl.MplugOwlTokenizer.from_pretrained(
            self.args.model_base
        )
        self.processor = self.mplug_owl.processing_mplug_owl.MplugOwlProcessor(
            image_processor, tokenizer
        )
        peft_config = LoraConfig(target_modules=r'.*language_model.*\.(q_proj|v_proj)', inference_mode=True, r=8,lora_alpha=32, lora_dropout=0.05)
        model = get_peft_model(model_base, peft_config)
        prefix_state_dict = torch.load(self.model_path, map_location='cpu')
        incompatible_keys = model.load_state_dict(prefix_state_dict, strict=False)
        assert all("inv_freq" in k for k in incompatible_keys.unexpected_keys), "only inv_freq buffer parameters should be in unexpected"
        model = model.merge_and_unload()
        self.model = model.cuda()

    def format_prompt(self, prompt):
        self.qs = ["The following is a conversation between a curious human and AI. The AI gives helpful, detailed, and polite answers to the human's questions.\nHuman: <image>\nHuman: " + prompt + "\nAI:"]
        return None

    def tokenize_prompt(self, formatted_prompt):
        if formatted_prompt is not None:
            raise ValueError("formatted_prompt must be None for mPLUGOwl")
        return None

    def generate_batch(self, input_ids, imgs):
        inputs = self.processor(
            images=imgs,
            text=self.qs,
            return_tensors='pt'
        ).to(self.model.device)
        # print(inputs.keys())
        inputs["input_ids"] = inputs["input_ids"].expand(len(imgs), -1)
        inputs["attention_mask"] = inputs["attention_mask"].expand(len(imgs), -1)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                # stopping_criteria=[stopping_criteria],
        )
        
        batch_outputs = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_outputs = [o.strip() for o in batch_outputs]
        return batch_outputs
        

class LLaMAAdapter(Evaluatee):
    def __init__(self, args):
        super().__init__(args)
        import llama
        
        self.llama = llama
        self.model_name = args.model_name
        self.llama_model_path = args.llama_model_path
    
    def load(self):
        self.model, self.processor = self.llama.load(self.model_name, self.llama_model_path, device="cuda")
        self.model.eval()

    def format_prompt(self, prompt):
        self.prompt = self.llama.format_prompt(prompt)

    def tokenize_prompt(self, formatted_prompt):
        if formatted_prompt is not None:
            raise ValueError("formatted_prompt must be None for LLaMAAdapter")
        return None
    
    def generate_batch(self, input_ids, imgs_list):
        imgs = []
        for img in imgs_list:
            imgs.append(self.processor(img).to("cuda"))
        imgs = torch.stack(imgs)
        with torch.inference_mode():
            output_ids = self.model.generate(
                imgs, [self.prompt for _ in range(len(imgs))], temperature=0.0, max_gen_len=1024)
        batch_outputs = [output.strip() for output in output_ids]
        return batch_outputs


class MiniGPT(Evaluatee):
    def __init__(self, args):
        # FIXME: convert args
        super().__init__(args)
        self.args.model_base = None
        self.args.do_sample = False
        self.args.queries = [0]
        self.args.conv_mode = None
        self.args.bs = self.args.per_device_batch_size
        self.args.overwrite = True
        self.args.num_ret = 1
        self.args.dev = False
        self.args.load8bit = False
        self.args.options = []
        self.gpu_id = int(os.environ['LOCAL_RANK']) % torch.cuda.device_count()
        self.device_str = 'cuda:{}'.format(self.gpu_id)

        import minigpt4.common.config
        import minigpt4.common.registry
        import minigpt4.conversation.conversation
        # from minigpt4.common.config import Config
        # from minigpt4.common.registry import registry
        # from minigpt4.conversation.conversation import CONV_VISION_Vicuna0, CONV_VISION_LLama2  #, StoppingCriteriaSub
        self.minigpt4 = minigpt4
        self.cfg = minigpt4.common.config.Config(self.args)
        conv_dict = {'pretrain_vicuna0': self.minigpt4.conversation.conversation.CONV_VISION_Vicuna0,
                    'pretrain_llama2': self.minigpt4.conversation.conversation.CONV_VISION_LLama2,
                    'pretrain': self.minigpt4.conversation.conversation.CONV_VISION_minigptv2}  # minigpt2

        self.conv = conv_dict[self.cfg.model_cfg.model_type].copy()

    def load(self):
        registry = self.minigpt4.common.registry.registry

        model_config = self.cfg.model_cfg
        model_config.device_8bit = self.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device_str)  # (LOCAL_RANK)


        key = list(self.cfg.datasets_cfg.keys())[0]
        vis_processor_cfg = self.cfg.datasets_cfg.get(key).vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model.eval()

    def format_prompt(self, prompt):
        qs = prompt
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], '<Img><ImageHere></Img> {}'.format(qs))
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    def tokenize_prompt(self, formatted_prompt):
        return formatted_prompt  # the model deals with it

    def process_outputs(self, outputs, conv):
        batch_outputs = [output.split(conv.sep)[0].replace("\u200b", "").strip() for output in outputs]
        return batch_outputs

    def generate_batch(self, input_ids, imgs):
        conv = self.conv.copy()
        stop_words_ids = [[835], [2277, 29937]]
        images = torch.stack([self.vis_processor(img) for img in imgs])
        with torch.inference_mode():
            outputs = self.model.generate(
                images.cuda(),
                [input_ids for _ in range(len(images))],
                do_sample=True,
                max_new_tokens=1024,
                # use_cache=True,
                stop_words_ids=stop_words_ids,
        )
        batch_outputs = self.process_outputs(outputs, conv)
        return batch_outputs


class MiniGPTv2(MiniGPT):
    def __init__(self, args):
        super().__init__(args)
        # setup correct conv
        del self.conv
        self.conv = self.minigpt4.conversation.conversation.CONV_VISION_minigptv2.copy()
        self.conv.system = "" # this may be done by default
    
    def process_outputs(self, outputs, conv):
        batch_outputs = [output.replace("\u200b", "").strip() for output in outputs]
        return batch_outputs


class Otter(Evaluatee):
    def __init__(self, args):
        super().__init__(args)
        import otter_ai
        self.otter_ai = otter_ai
    
    def load(self):
        from transformers import CLIPImageProcessor
        self.model = self.otter_ai.OtterForConditionalGeneration.from_pretrained(self.args.model_path, torch_dtype=torch.float16).cuda()
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.model.eval()

    def format_prompt(self, prompt):
        self.prompt = f"<image>User: {prompt} GPT:<answer>"
        return None

    def tokenize_prompt(self, formatted_prompt):
        if formatted_prompt is not None:
            raise ValueError("formatted_prompt must be None for Otter")
        return None
    
    def generate_batch(self, input_ids, imgs):
        vision_x = self.image_processor.preprocess(imgs, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(1)
        lang_x = self.model.text_tokenizer(
            [
                self.prompt,
            ],
            return_tensors="pt",
        )
        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]
        with torch.inference_mode():
            output_ids = self.model.generate(
                vision_x=vision_x.to(self.model.device),
                lang_x=lang_x_input_ids.expand(len(imgs), -1).to(self.model.device),
                attention_mask=lang_x_attention_mask.expand(len(imgs), -1).to(self.model.device),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                # num_return_sequences=args.num_ret,
                pad_token_id=self.model.text_tokenizer.eos_token_id,
                # stopping_criteria=[stopping_criteria],
        )
        batch_outputs = self.model.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        batch_outputs = [o.split("GPT:")[-1].strip() for o in batch_outputs]
        return batch_outputs


MODELS = dict(
    LLaVA=LLaVAMistralVicuna,
    MiniGPT=MiniGPT,
    MiniGPTv2=MiniGPTv2,
    InstructBLIP=InstructBLIP,
    BLIP2=BLIP2,
    mPLUGOwl=mPLUGOwl,
    LRVInstruct=LRVInstruct,
    LLaMAAdapter=LLaMAAdapter,
    Otter=Otter,
)


def get_evaluated_model(name, args):
    assert name in MODELS
    return MODELS[name](args)

