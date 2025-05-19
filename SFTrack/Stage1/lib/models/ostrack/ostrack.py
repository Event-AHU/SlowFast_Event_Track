"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.ostrack.vit_ce_fast import vit_base_patch16_224_ce_fast
from lib.utils.box_ops import box_xyxy_to_cxcywh

from torch_geometric import data
from ..fast_gcn import GraphRes, GraphRes_Fast

class Slow(nn.Module):
    """ This is the base class for OSTrack """
    
    def __init__(self, transformer, gcn, box_head, aux_loss=False, head_type="CORNER", training=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.gcn = gcn
        self.box_head = box_head
        
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.training = training    
        self.template_gcn_feat = None

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_bin_event,
                search_bin_event,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                return_multi_results=False,
                ):

        if self.training or self.template_gcn_feat is None:
            if template_bin_event is not None:
                self.template_gcn_feat = self.gcn_forward(data=template_bin_event.clone())

        if self.template_gcn_feat is not None and search_bin_event is not None:
            search_gcn_feat = self.gcn_forward(data=search_bin_event.clone())
            x = self.backbone(z=template, x=search, 
                            template_gcn_feat=self.template_gcn_feat,
                            search_gcn_feat=search_gcn_feat,
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=return_last_attn, )
    
        else:    
            x = self.backbone(z=template, x=search, 
                            ce_template_mask=ce_template_mask,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=return_last_attn,)
                
        feat_last = x
        out = self.forward_head(feat_last, None)
    
        return out
   
    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def gcn_forward(self, data: data.Batch) -> torch.Tensor:
        x = self.gcn.forward(data)
        
        return x

class Fast(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, fast_vit, fast_gcn, fast_head, aux_loss=False, head_type="CORNER", training=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.fast_vit = fast_vit
        self.fast_gcn = fast_gcn
        self.fast_head = fast_head
        
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(fast_head.feat_sz)
            self.feat_len_s = int(fast_head.feat_sz ** 2)

        if self.aux_loss:
            self.fast_head = _get_clones(self.fast_head, 6)
        
        self.training = training    
        self.template_gcn_feat = None

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                template_bin_event,
                search_bin_event,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                return_multi_results=False,
                ):

        x = self.fast_vit(z=template, x=search)
        
        if template_bin_event is None:
            out = self.forward_fast_head(x, None)
            return out
        else:
            if self.training or self.template_gcn_feat is None:
                self.template_gcn_feat = self.gcn_forward(data=template_bin_event.clone())

        if self.template_gcn_feat is not None and search_bin_event is not None: 
            if return_multi_results: ## generate multi results
                out_list = []   
                for sub_search_bin_event in search_bin_event:
                    sub_search_bin_event = self.gcn_forward(data=sub_search_bin_event.clone())
                    fusion_gcn_feat = torch.cat((self.template_gcn_feat.unsqueeze(1).expand(-1, 64, -1), sub_search_bin_event.unsqueeze(1).expand(-1, 256, -1)), dim=1)
                    fusion_out = x * fusion_gcn_feat + x
                    out = self.forward_fast_head(fusion_out, None)
                    out_list.append(out)
                return out_list
            else: ## generate single result
                search_gcn_feat = self.gcn_forward(data=search_bin_event.clone()) 
                fusion_gcn_feat = torch.cat((self.template_gcn_feat.unsqueeze(1).expand(-1, 64, -1), search_gcn_feat.unsqueeze(1).expand(-1, 256, -1)), dim=1)
                fusion_out = x * fusion_gcn_feat + x
                out = self.forward_fast_head(fusion_out, None)
                return out
        else:
            return self.forward_fast_head(x, None)
   
    def forward_fast_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.fast_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.fast_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def gcn_forward(self, data: data.Batch) -> torch.Tensor:
        x = self.fast_gcn.forward(data)
        
        return x
        
def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
        
    if cfg.TRAIN.TRACKER_TYPE == "Slow_Tracker":
        if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
            backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )
            hidden_dim = backbone.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError
        
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head = build_box_head(cfg, hidden_dim)
        
        dataset=cfg.DATA.TRAIN.DATASETS_NAME[0]
        if dataset == 'EventVOT':
            input_shape=torch.tensor((1280, 720, 3), device='cuda')
        elif dataset == 'FE240':
            input_shape=torch.tensor((346, 260, 3), device='cuda')
        elif dataset == 'COESOT':
            input_shape=torch.tensor((346, 260, 3), device='cuda')
        else:
            NotImplementedError("Error dataset ~!")
            
        gcn = GraphRes(dataset='event_data', input_shape=input_shape, num_outputs=768, bias=True)
        
        model = Slow(
            backbone,
            gcn,
            box_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
            training=training,
        )
        
    elif cfg.TRAIN.TRACKER_TYPE == "Fast_Tracker":
        if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
            fast_vit = vit_base_patch16_224_ce_fast(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )
            hidden_dim = fast_vit.embed_dim
            patch_start_index = 1
        else:
            raise NotImplementedError

        fast_vit.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        fast_head = build_box_head(cfg, hidden_dim)

        dataset=cfg.DATA.TRAIN.DATASETS_NAME[0]
        if dataset == 'EventVOT':
            input_shape=torch.tensor((1280, 720, 3), device='cuda')
        elif dataset == 'FE240':
            input_shape=torch.tensor((346, 260, 3), device='cuda')
        elif dataset == 'COESOT':
            input_shape=torch.tensor((346, 260, 3), device='cuda')
        else:
            NotImplementedError("Error dataset ~!")
            
        fast_gcn = GraphRes_Fast(dataset='event_data', input_shape=input_shape, num_outputs=768, bias=True)

        model = Fast(
            fast_vit,
            fast_gcn,
            fast_head,
            aux_loss=False,
            head_type=cfg.MODEL.HEAD.TYPE,
            training=training,
        )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model


