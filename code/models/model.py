import torch.nn as nn
import numpy as np
from code.models.cspdarknet import CspDarkNet
from code.models.lstm_sa import LSTM_SA
from code.models.mcn_heads import MCNhead
from code.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention, ImageFusion


class GVGNet_KD(nn.Module):
    def __init__(self, pretrained_emb, token_size, cfg):
        super(GVGNet_KD, self).__init__()
        self.visual_encoder = CspDarkNet(pretrained=cfg["model"]["visual_pretrained"],
                                         pretrained_weight_path=cfg["model"]["visual_backbone_path"],
                                         freeze_backbone=False,
                                         multi_scale_outputs=True)

        self.lang_encoder = LSTM_SA(depth=3,
                                    hidden_size=512,
                                    num_heads=1,
                                    ffn_size=2048,
                                    flat_glimpses=1,
                                    word_embed_size=300,
                                    pretrained_emb=pretrained_emb,
                                    token_size=token_size,
                                    dropout_rate=0.0,
                                    freeze_embedding=True,
                                    use_glove=True)

        self.fusion_manner = SimpleFusion(v_planes=1024, q_planes=512, out_planes=1024)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), scaled=True)

        self.det_garan = GaranAttention(d_q=512, d_v=512)
        self.seg_garan = GaranAttention(d_q=512, d_v=512)

        self.head = MCNhead(hidden_size=512,
                            anchors=[[137, 256], [248, 272], [386, 271]],
                            arch_mask=[[0, 1, 2]],
                            layer_no=0,
                            in_ch=512,
                            n_classes=0,
                            ignore_thre=0.5)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, img, ref, gaze_feat, det_label=None, seg_label=None):
        # img [N, 3, 416, 416]
        # ref [N, 3]
        # gaze_feat [N, 512, 7, 7]

        x = self.visual_encoder(img)    # [N, 256, 52, 52] [N, 512, 26, 26] [N, 1024, 13, 13]
        y = self.lang_encoder(ref)      # [N, 2, 512]

        x[-1] = self.fusion_manner(x[-1], y['flat_lang_feat'], gaze_feat)
        bot_feats, mid_feats, top_feats = self.multi_scale_manner(x)

        # bot_feats = bot_feats.cpu().numpy()
        # np.save("bot_feats.npy", bot_feats)

        bot_feats, seg_map, seg_attn = self.seg_garan(y['flat_lang_feat'], bot_feats)
        top_feats, det_map, det_attn = self.det_garan(y['flat_lang_feat'], top_feats)

        # top torch.Size([2, 512, 13, 13])
        # mid torch.Size([2, 512, 26, 26])
        # bot torch.Size([2, 512, 52, 52])

        if self.training:
            loss, loss_det, loss_seg = self.head(top_feats, mid_feats, bot_feats, det_label, seg_label, det_map, seg_map,
                                                 det_attn, seg_attn)
            return loss, loss_det, loss_seg
        else:
            box, mask = self.head(top_feats, mid_feats, bot_feats)
            return box, mask
