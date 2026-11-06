# Models/RadFM/module/utils.py
"""
本文件 = 两部分合并，来源分别标注：
1) 官方 src/Model/RadFM/utils.py 原样内容
   （PerceiverResampler / Flamingo 组件 / get_visual_encoder 等）
   来源：https://github.com/chaoyi-wu/RadFM/blob/main/src/Model/RadFM/utils.py
2) 复现文本端工具（get_tokenizer / build_prompt / encode_example /
   build_vision_x / batch_to_inputs / PHASE_DESCS）
   - get_tokenizer 逐行复制自官方 Quick_demo/test.py::get_tokenizer
     https://github.com/chaoyi-wu/RadFM/blob/main/Quick_demo/test.py
   - 占位符格式 "<image>+<image32i..32i+31>+</image>" <- 官方 combine_and_preprocess
   - vision_x 约定 (B,S,C,H,W,D)、每图 32 个视觉 token <- 官方 my_embedding_layer.py
注意：官方 demo 的 combine_and_preprocess 用 torch.cat(images, dim=1)，仅单图时与
(B,S,C,H,W,D) 等价；多图会把通道维拼坏，本复现不使用该函数。
"""
from __future__ import annotations
from typing import Sequence
import torch
from torch import einsum, nn
from transformers import LlamaTokenizer
from .blocks import ModifiedResNet, PMC_CLIP_cfg
from torchvision import transforms
from PIL import Image
from einops import rearrange, repeat
from einops_exts import rearrange_many

"""
文本端工具
"""
# 期相说明文本：prompt 中标注每张图是什么（帮助模型条件化）
PHASE_DESCS = {
    "NC": "non-contrast phase",
    "Arterial": "arterial phase",
    "Venous": "venous phase",
    "Delay": "delay phase",
}
DEFAULT_INSTRUCTION = "Please caption this scan with findings."


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    """【源自官方 Quick_demo/test.py::get_tokenizer】
    [复现适配] 原实现把 add_special_tokens 放在循环内调用 100 次且列表累积，
    重复注册约 16 万次 token（O(n^2)），启动慢；改为收集齐全后一次注册。
    token 的添加顺序与官方完全一致（<image>=32000, </image>=32001, <image0>... 顺序递增），
    故 id 分配逐位相同，下游 one-hot 索引对齐不受影响。"""
    text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    all_special = ["<image>", "</image>"] + [
        f"<image{i * image_num + j}>"
        for i in range(max_img_size) for j in range(image_num)
    ]
    text_tokenizer.add_special_tokens({"additional_special_tokens": all_special})
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2
    image_padding_tokens = [
        "".join(f"<image{i * image_num + j}>" for j in range(image_num))
        for i in range(max_img_size)
    ]
    return text_tokenizer, image_padding_tokens


def build_prompt(phases: Sequence[str], image_padding_tokens,
                 instruction: str = DEFAULT_INSTRUCTION) -> str:
    """指令段：每张图前插占位符+期相说明，结尾接报告生成指令。"""
    parts = []
    for i, ph in enumerate(phases):
        parts.append(f"<image>{image_padding_tokens[i]}</image>{PHASE_DESCS[ph]}.")
    parts.append(instruction)
    return " ".join(parts)


def encode_example(tokenizer, image_padding_tokens, phases, response,
                   max_len: int = 2048, instruction: str = DEFAULT_INSTRUCTION) -> dict:
    """编码单例：prompt 段 labels 置 -100（对应论文"指令不计 loss"，官方 forward
    返回 HF 内部 labels loss -> src/Model/RadFM/multimodality_model.py），
    仅 response(+EOS) 段参与 loss。BOS 由 LlamaTokenizer 默认添加（同官方 demo）。"""
    prompt = build_prompt(phases, image_padding_tokens, instruction)
    p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if response:   # [复现适配] 空 response(推理)不追加 EOS：训练时 EOS 后接 PAD，
        r_ids = r_ids + [tokenizer.eos_token_id]   # 推理若 prompt 后直接跟 EOS 会诱导模型输出 <unk>
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    lang_x = torch.tensor([ids], dtype=torch.long)
    return {
        "lang_x": lang_x,
        "labels": torch.tensor([labels], dtype=torch.long),
        "attention_mask": torch.ones_like(lang_x),
        "loss_reweight": torch.ones_like(lang_x, dtype=torch.float),  # 官方返回值不含权重
    }


"""
影像端工具
"""
def build_vision_x(x: torch.Tensor) -> torch.Tensor:
    """x: (B, S, H, W, D) float [0,1]（MultiphaseReport 输出）-> (B, S, 3, H, W, D)。
    官方 ViT 输入通道为 3（patch_dim=3*32*32*4，src/Model/RadFM/vit_3d.py channels=3），
    单通道 CT 复制为 3 通道以匹配预训练 patch embedding 权重。"""
    return x.unsqueeze(2).expand(-1, -1, 3, -1, -1, -1).contiguous()


def batch_to_inputs(batch, tokenizer, image_padding_tokens, phases,
                    max_len: int = 2048, instruction: str = DEFAULT_INSTRUCTION,
                    response=None) -> dict:
    """把 multiphase_collate 的输出（B 必须为 1）转成 LiverRadFM.forward 的输入 dict。
    response=None(默认): 用 batch["text"][0] 作 response（训练/算nll）；
    response=""（推理）: 只保留 prompt+EOS，不偷看 GT。"""
    assert batch["x"].shape[0] == 1, "RadFM 3D 训练/推理固定 batch=1"
    enc = encode_example(tokenizer, image_padding_tokens, phases,
                         batch["text"][0] if response is None else response,
                         max_len, instruction)
    enc["vision_x"] = build_vision_x(batch["x"])
    return enc


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1:])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def get_visual_encoder(model_str):
    """
    Args:
        str (_type_): str_to_model_path
    Return:
        vision_model, visual_dim, img_preprocessor
    """
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_preprocessor = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if 'PMC-CLIP' in model_str:
        # vision_cfg = json.load(open(model_args.visual_model_config,'r'))['vision_cfg']
        vision_cfg = PMC_CLIP_cfg()
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        vision_model = ModifiedResNet(
            layers=vision_cfg.layers,
            heads=vision_heads,
            output_dim=768,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width
        )
        vision_model = vision_load_pretrain(vision_model, model_str)
        vision_model = nn.Sequential(*list(vision_model.children())[:-2])
        visual_dim = 1024
    return vision_model, visual_dim, img_preprocessor


def vision_load_pretrain(resnet, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.visual.', ''): v for k, v in state_dict.items() if '.visual' in k}
    resnet.load_state_dict(state_dict)
    return resnet


"""
影像端perceiver
Taken from https://github.com/lucidrains/flamingo-pytorch
"""
def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth=6,
            dim_head=64,
            heads=8,
            num_latents=64,
            max_num_media=None,
            max_num_frames=None,
            ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_visual,
            dim_head=64,
            heads=8,
            only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, attend_previous=True):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            attend_previous: bool
                If false, ignores immediately preceding image and starts attending when following image
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            # at each boolean of True, increment the time counter (relative to media time)
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(T_img, device=x.device) + 1

            if not attend_previous:
                text_time[~media_locations] += 1
                # make sure max is still the number of images in the sequence
                text_time[
                    text_time
                    > repeat(
                        torch.count_nonzero(media_locations, dim=1),
                        "b -> b i",
                        i=text_time.shape[1],
                    )
                    ] = 0

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_visual,
            dim_head=64,
            heads=8,
            ff_mult=4,
            only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
            self,
            x,
            media,
            media_locations=None,
            attend_previous=True,
    ):
        x = (
                self.attn(
                    x,
                    media,
                    media_locations=media_locations,
                    attend_previous=attend_previous,
                )
                * self.attn_gate.tanh()
                + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x
