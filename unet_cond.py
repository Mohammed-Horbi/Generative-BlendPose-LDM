import torch
from einops import einsum
import torch.nn as nn
import torch.nn.functional as F
from blocks import get_time_embedding
from blocks import DownBlock, MidBlock, UpBlockUnet
# from utils.config_utils import *


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """

    def __init__(self, im_channels, use_up=True):
        super().__init__()
        self.down_channels = [ 320, 640, 1280, 1280 ]  # Increased for better feature extraction
        self.mid_channels = [ 1280, 1280 ]  # Keeping consistent high-dim features
        self.t_emb_dim = 1280  # Matching feature dimensions
        self.down_sample = [ True, True, True ]
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        self.attns = [True, True, True]
        self.norm_channels = 32
        self.num_heads = 16
        self.conv_out_channels = 128

        # Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        ######## Class, Mask and Text Conditioning Config #####
        self.class_cond = False
        self.text_cond = True
        self.image_cond = False
        # self.text_embed_dim = None
        self.text_embed_dim = 768
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.text_cond # or self.image_cond or self.class_cond
        # self.condition_config = get_config_value(model_config, 'condition_config', None)
        # if self.condition_config is not None:
        #     assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
        #     condition_types = self.condition_config['condition_types']
        #     # if 'class' in condition_types:
        #     #     validate_class_config(self.condition_config)
        #     #     self.class_cond = True
        #     #     self.num_classes = self.condition_config['class_condition_config']['num_classes']
        #     # if 'text' in condition_types:
        #     #     validate_text_config(self.condition_config)
        #     #     self.text_cond = True
        #     #     self.text_embed_dim = 768 #self.condition_config['text_condition_config']['text_embed_dim']
        #     # if 'image' in condition_types:
        #     #     self.image_cond = True
        #     #     self.im_cond_input_ch = self.condition_config['image_condition_config'][
        #     #         'image_condition_input_channels']
        #     #     self.im_cond_output_ch = self.condition_config['image_condition_config'][
        #     #         'image_condition_output_channels']
        # # if self.class_cond:
        # #     # Rather than using a special null class we dont add the
        # #     # class embedding information for unconditional generation
        # #     self.class_emb = nn.Embedding(self.num_classes,
        # #                                   self.t_emb_dim)

        # # if self.image_cond:
        # #     # Map the mask image to a N channel image and
        # #     # concat that with input across channel dimension
        # #     self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
        # #                                   out_channels=self.im_cond_output_ch,
        # #                                   kernel_size=1,
        # #                                   bias=False)
        # #     self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch,
        # #                                     self.down_channels[0], kernel_size=3, padding=1)
        # else:
        #     self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        # self.cond = self.text_cond # or self.image_cond or self.class_cond
        ###################################

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.GELU(),  # Using GELU activation for better performance
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.Dropout(0.1)
        )

        
        self.up_sample = list(reversed(self.down_sample))
        self.downs = nn.ModuleList([])

        # Build the Downblocks
        for i in range(len(self.down_channels) - 1):
            # Cross Attention and Context Dim only needed if text condition is present
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels,
                                        cross_attn=self.text_cond,
                                        context_dim=self.text_embed_dim))

        self.mids = nn.ModuleList([])
        # Build the Midblocks
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      cross_attn=self.text_cond,
                                      context_dim=self.text_embed_dim))

        self.ups = nn.ModuleList([])
        if use_up:
            # Build the Upblocks
            for i in reversed(range(len(self.down_channels) - 1)):
                self.ups.append(
                    UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                                self.t_emb_dim, up_sample=self.down_sample[i],
                                num_heads=self.num_heads,
                                num_layers=self.num_up_layers,
                                norm_channels=self.norm_channels,
                                cross_attn=self.text_cond,
                                context_dim=self.text_embed_dim))

            self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
            self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond_txt=None, cond_img=None, text_weight=0.65, image_weight=0.35):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        if self.cond:
            assert cond_txt is not None, \
                "Model initialized with conditioning so cond_input cannot be None"
        # if self.image_cond:
        #     ######## Mask Conditioning ########
        #     validate_image_conditional_input(cond_input, x)
        #     im_cond = cond_input['image']
        #     im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
        #     im_cond = self.cond_conv_in(im_cond)
        #     assert im_cond.shape[-2:] == x.shape[-2:]
        #     x = torch.cat([x, im_cond], dim=1)
        #     # B x (C+N) x H x W
        #     out = self.conv_in_concat(x)
        #     #####################################
        # else:
            # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W

        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        # ######## Class Conditioning ########
        # if self.class_cond:
        #     validate_class_conditional_input(cond_input, x, self.num_classes)
        #     class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
        #     t_emb += class_embed
        # ####################################

        context_hidden_states = None
        # if self.text_cond:
        #     assert 'text' in cond_input, \
        #         "Model initialized with text conditioning but cond_input has no text information"
        ##########################################################
        # Text Conditioning projection
        # cond_txt = cond_txt + self.text_proj(cond_txt)  # (B, 77, 768) For adding residual connection (x + self.text_proj(x))
        ##########################################################
        # cond_txt = cond_txt / (cond_txt.norm(dim=-1, keepdim=True) + 1e-6)
        if cond_img is not None:
            # Process image hidden states
            img_emb = cond_img.transpose(1, 2)  # (B, 1024, 257) Output of the image clipping model
            # print("img_emb after first transpose shape: ", img_emb.shape)
            img_emb = F.interpolate(img_emb, size=77, mode='linear', align_corners=False)  # (B, 1024, 77)
            img_emb = img_emb.transpose(1, 2)  # (B, 77, 1024)
            # print("img_emb after second transpose shape: ", img_emb.shape)
            # Apply image projection
            img_emb = self.image_proj(img_emb)  # (B, 77, 768)
            # print("img_emb after image_proj calling shape: ", img_emb.shape)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-6)

            cond_txt = self.text_proj(cond_txt)  # (B, 77, 768) For adding residual connection (x + self.text_proj(x))
            cond_txt = cond_txt / (cond_txt.norm(dim=-1, keepdim=True) + 1e-6)
            # print("cond_txt after  text_proj calling shape: ", cond_txt.shape)
            # Weighted fusion
            # context_hidden_states = text_weight * cond_txt.squeeze(1) + image_weight * img_emb                    
            context_hidden_states = text_weight * cond_txt + image_weight * img_emb
            # print("context_hidden_states after weighting text and image shape: ", context_hidden_states.shape)
        else:
            # cond_txt = self.text_proj(cond_txt)  # (B, 77, 768) For adding residual connection (x + self.text_proj(x))
            cond_txt = cond_txt + self.text_proj(cond_txt)  # (B, 77, 768) For adding residual connection (x + self.text_proj(x))
            context_hidden_states = cond_txt / (cond_txt.norm(dim=-1, keepdim=True) + 1e-6)
            # context_hidden_states = cond_txt # Here is the text embeddings
            
        down_outs = []

        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4

        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
        # out B x C3 x H/4 x W/4

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
