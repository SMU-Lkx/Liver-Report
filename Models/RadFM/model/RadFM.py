# Models/RadFM/model/RadFM.py
"""
LiverRadFM：RadFM 在肝脏多期相 CT 报告生成上的复现封装。

来源对照（官方仓库 https://github.com/chaoyi-wu/RadFM）：
- 模型本体 MultiLLaMAForCausalLM <- src/Model/RadFM/multimodality_model.py
  https://github.com/chaoyi-wu/RadFM/blob/main/src/Model/RadFM/multimodality_model.py
- 权重加载 <- Quick_demo/test.py::main（pytorch_model.bin 全量 state_dict）
- forward/generate 调用方式 <- 同上 multimodality_model.py 与 Quick_demo/test.py
- [复现适配] 冻结视觉编码器 + LLM-LoRA：官方训练为全参微调，1633 例数据下改用 LoRA
- [复现适配] bf16 加载：fp32 权重 ~56GB 超出显存预算
"""
from __future__ import annotations
import torch
from torch import nn
from Models.RadFM.model.multimodality_model import MultiLLaMAForCausalLM


class LiverRadFM(nn.Module):
    """输入约定见 RadFM_utils.batch_to_inputs；loss 为 response 段 CE。"""
    def __init__(self, cfg: dict):
        super().__init__()
        device = cfg.get("device")
        if device is None:
            device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
        self.device = torch.device(device)

        # ---- 加载官方模型与权重（同 Quick_demo/test.py::main）----
        self.model = MultiLLaMAForCausalLM(
            lang_model_path=cfg["lang_model_path"],
            kebert_path=cfg["kebert_path"],
        )
        ckpt = torch.load(cfg["ckpt"], map_location="cpu")
        # [复现适配] 官方 ckpt 由旧版 transformers 保存，含新版已改为"现算"的确定性 buffer：
        #   rotary_emb.inv_freq          —— LLaMA 旋转位置编码，值由 config 现算，非学习参数
        #   bert embeddings.position_ids —— 同上，非持久化 buffer
        # 剔除后仍用 strict=True 加载，避免 strict=False 掩盖真实的 key 缺失。
        ckpt = {k: v for k, v in ckpt.items()
                if not k.endswith("rotary_emb.inv_freq")
                and k != "embedding_layer.bert_model.embeddings.position_ids"}
        self.model.load_state_dict(ckpt, strict=True)
        del ckpt   # 权重已注入模型，释放 CPU 侧 ~51GB 内存

        # ---- [复现适配] 冻结视觉编码器（embedding_layer = 3D ViT + Perceiver + 投影）----
        for p in self.model.embedding_layer.parameters():
            p.requires_grad_(False)

        # ---- [复现适配] LLM 注入 LoRA（官方为全参训练；1633 例数据下用 LoRA）----
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],  # 参数全部来自 YAML
            lora_dropout=cfg["lora_dropout"], bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.model.lang_model = get_peft_model(self.model.lang_model, lora_cfg)
        self.model.lang_model.print_trainable_parameters()

        # ---- [复现适配] bf16（fp32 ~56GB 放不下）----
        self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()

    # ---------------- 训练 ----------------
    def forward(self, inputs: dict):
        out = self.model(
            lang_x=inputs["lang_x"].to(self.device),
            vision_x=inputs["vision_x"].to(self.device, dtype=torch.bfloat16),
            attention_mask=inputs["attention_mask"].to(self.device),
            labels=inputs["labels"].to(self.device),
            loss_reweight=inputs["loss_reweight"].to(self.device),
            key_words_query=None,
        )
        # 注意：官方 forward 返回的 logits 实为自定义的序列级准确率（见官方源码）
        return out["loss"], out["logits"]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def generate(self, inputs: dict, max_new_tokens: int = 512,
                 num_beams: int = 1, repetition_penalty: float = 1.2) -> torch.Tensor:
        """返回生成的新 token（不含 prompt）。生成方式同官方 MultiLLaMAForCausalLM.generate，
        但解除官方写死的 max_new_tokens=200 / top_k=50。"""
        self.model.embedding_layer.flag = "Text"
        input_embedding, _ = self.model.embedding_layer(
            inputs["lang_x"].to(self.device),
            inputs["vision_x"].to(device=self.device, dtype=torch.bfloat16))
        gen = self.model.lang_model.generate(
            inputs_embeds=input_embedding,
            attention_mask=inputs["attention_mask"].to(self.device),
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            eos_token_id=2, pad_token_id=0)
        # [复现适配] transformers>=4.4x 中仅传 inputs_embeds 的 generate 只返回新生成 token；
        # 旧版返回 prompt+新 token。用"前缀是否等于输入 ids"确定性判别，两种版本都安全。
        L = inputs["lang_x"].shape[1]
        if gen.shape[1] > L and torch.equal(gen[0, :L].cpu(), inputs["lang_x"][0].cpu()):
            return gen[:, L:]
        return gen


class LoRAAdapterShim(nn.Module):
    """[RadFM复现适配] 把 PeftModel 的 adapter 参数伪装成独立模块，
    使 CkptManager 只保存/恢复 LoRA 权重（约几十MB），而非 14B 全量 state_dict。
    保存/恢复逻辑与 CT2Rep/MRI2Rep 完全共用 CkptManager，无二套实现。"""

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        from peft.utils import get_peft_model_state_dict
        return get_peft_model_state_dict(self.peft_model)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # [复现适配] peft 存/取 key 前缀约定随版本漂移，官方 loader 不可依赖：
        # 将快照 key 归一化映射到活模型的实际命名，加载后逐 key 校验，杜绝静默失败。
        # [复现适配] peft 存档命名与活模型命名存在系统性差异，这里显式修补两种：
        #   ① adapter 名作用域: 存档 "...lora_A.weight"  <-> 活模型 "...lora_A.default.weight"
        #   ② base_model 前缀有无两种组合
        # 映射不到直接报错，加载后逐 key 校验值，杜绝静默失败。
        live = self.peft_model.state_dict()
        fixed = {}
        for k, v in state_dict.items():
            cands = [k]
            if ".lora_" in k and k.endswith(".weight"):
                cands.append(k[:-len(".weight")] + ".default.weight")   # ①
            cands += [c[len("base_model.model."):] if c.startswith("base_model.model.")  # ②
                      else "base_model.model." + c for c in list(cands)]
            tgt = next((c for c in cands if c in live), None)
            if tgt is None:
                raise RuntimeError(f"adapter key 无法映射到活模型命名: {k}")
            fixed[tgt] = v
        self.peft_model.load_state_dict(fixed, strict=False)
        new_live = self.peft_model.state_dict()
        for tgt, v in fixed.items():
            if not torch.allclose(new_live[tgt].float().cpu(), v.float().cpu(), atol=1e-5):
                raise RuntimeError(f"adapter 注入校验失败(值不一致): {tgt}")
