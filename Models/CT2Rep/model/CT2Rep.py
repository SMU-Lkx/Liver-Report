# Models/CT2Rep/model/CT2Rep.py
"""CT2Rep baseline 模型 .

结构 = CTViT(spatial+causal transformer) -> AvgPool+fc -> R2Gen EncoderDecoder(RM+MCLN)
接口与 MRI2RepTextGen 对齐: forward(x, y) 吃全量 y (含BOS..EOS), 内部移位;
                           generate(x, max_len) 贪心解码.
输入约定: x (B,4,H,W,D) float [0,1] (缓存原样), y (B,L) long, PAD=0 BOS=1 EOS=2.
"""
import torch
import torch.nn as nn
from Models.CT2Rep.model.ctvit import CTViT
from Models.CT2Rep.modules.encoder_decoder import EncoderDecoder


class CharTokenizerAdapter:
    """CharTokenizer -> CT2Rep(R2Gen) tokenizer 接口 (att_model 需要 len(idx2token)).
    注意: 这不是神经网络, 只是把我们的词表包成官方代码期望的形状."""

    def __init__(self, tok):
        self.tok = tok
        self.idx2token = ["<pad>", "<bos>", "<eos>"] + [tok.inv[i] for i in sorted(tok.inv)]
        self.token2idx = {c: i for i, c in enumerate(self.idx2token)}

    def decode(self, ids):
        return self.tok.decode(ids)

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]


class CT2Rep(nn.Module):
    """CT2Rep: CTViT visual extractor + R2Gen encoder-decoder (RM + MCLN)."""

    def __init__(self, cfg, tok_adapter):
        super().__init__()
        self.max_seq_length = int(cfg.max_seq_length)

        self.ctvit = CTViT(
            dim=512,
            codebook_size=8192,       # 官方代码码本大小
            image_size=224,           # 我们: 224 (官方 480)
            patch_size=16,            # 我们: 16  (官方 24)  -> 空间 token 14x14
            temporal_patch_size=8,    # 我们: 8   (官方 12)  -> 深度 token 20
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
            channels=4,               # 四期相 (官方单通道 1)
            use_vgg_and_gan=False,    # 关键: 不构建 vgg16/discriminator (官方代码未传此参时会联网下权重)
        )
        # 官方 VisualExtractor 的等价实现, token 网格 20x14x14 (官方 20x20x20)
        self.avgpool = nn.AvgPool3d(kernel_size=(20, 14, 14))
        self.fc = nn.Linear(512, 512)
        self.encoder_decoder = EncoderDecoder(cfg, tok_adapter)

    def _encode(self, x):
        # x: (B,4,H,W,D) [0,1] -> CTViT 要 (B,C,D,H,W) 且 [-1,1]
        x = (x * 2 - 1).permute(0, 1, 4, 2, 3)
        tokens = self.ctvit(x, return_encoded_tokens=True)   # (B,20,14,14,512)
        patch = tokens.permute(0, 4, 1, 2, 3)                # (B,512,20,14,14)
        fc_feats = self.fc(self.avgpool(patch).flatten(1))   # (B,512)
        att_feats = patch.flatten(2).transpose(1, 2)         # (B,3920,512)
        return fc_feats, att_feats

    def forward(self, x, y):
        """x (B,4,H,W,D), y (B,L) 全量(含BOS). 返回 (B,L-1,V) log-probs, 与 y[:,1:] 对齐."""
        fc_feats, att_feats = self._encode(x)  # 肝脏全局特征，局部特征
        return self.encoder_decoder(fc_feats, att_feats, y, mode="forward")

    @torch.no_grad()
    def generate(self, x, max_len=None):
        fc_feats, att_feats = self._encode(x)
        seq, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")  # beam_size=1 -> greedy
        return seq                                                    # (B,L) ids, 含 BOS, decode 时剥掉


def build_generator(cfg, tok_adapter) -> CT2Rep:
    return CT2Rep(cfg, tok_adapter)
