# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from wan.modules.model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, output_dim, temporal_downsample=False):
        """
        ActionEncoder using 1D temporal convolution to aggregate actions.
        
        Args:
            action_dim: Dimension of each action vector
            hidden_dim: Hidden dimension for internal processing
            output_dim: Output dimension (should be 6 * model_dim for AdaLN-zero)
            num_frames: Number of output latent frames
        """
        super().__init__()
        
        # Action embedding MLP
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Use Video VAE-style temporal downsampling, the downsample rate is fixed 4x
        # 4x downsampling: stride=4, kernel=5, padding=2
        # Output: (T + 2*2 - 5) // 4 + 1 = (T - 1) // 4 + 1 (for valid T)
        if temporal_downsample:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU()
            )
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Zero-init for AdaLN-Zero strategy
        # At initialization, action path contributes 0, preserving pretrained behavior
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, actions):
        """
        Args:
            actions: (B, Ta, action_dim) - sequence of actions
        Returns:
            output: (B, num_frames, output_dim) - aggregated action features
        """
        
        x = self.action_embedding(actions)  # (B, Ta, hidden_dim)
        if hasattr(self, 'temporal_conv'):
            x = x.transpose(1, 2)
            x = self.temporal_conv(x)  # (B, hidden_dim, T')
            x = x.transpose(1, 2)
        output = self.output_proj(x)  # (B, num_frames, output_dim)
        return output


class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c, c_skip


class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x


class VaceWanActionModel(WanModel):

    @register_to_config
    def __init__(self,
                 action_dim=14,
                 action_hidden_dim=128,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        """
        Args:
            action_dim: Dimension of action vector
            action_hidden_dim: Hidden dimension for action encoder
            ... (other args from base WanModel)
        """
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim,
                         freq_dim, text_dim, out_dim, num_heads, num_layers,
                         window_size, qk_norm, cross_attn_norm, eps)

        self.action_dim = action_dim
        self.action_hidden_dim = action_hidden_dim
        
        self.action_encoder = ActionEncoder(
            action_dim=action_dim, 
            hidden_dim=action_hidden_dim, 
            output_dim=6*self.dim,
            temporal_downsample=False
        )

        self.vace_layers = [i for i in range(0, self.num_layers, 2)
                           ] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {
            i: n for n, i in enumerate(self.vace_layers)
        }

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=self.vace_layers_mapping[i]
                if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=i) for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        for block in self.vace_blocks:
            c, c_skip = block(c, **new_kwargs)
            hints.append(c_skip)
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        actions=None,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # if y is not None:
        #     x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # actions: (B, Ta, action_dim)
        if actions is not None:
            with amp.autocast(dtype=torch.float32):
                act_e = self.action_encoder(actions) # (B, num_latent_frames, dim)
                act_e = act_e.mean(dim=1)  # Aggregate over time: (B, dim)
                act_e = act_e.unflatten(-1, (6, self.dim)) 
                # Combined modulation (now global)
                e_total = e0 + act_e
        else:
            e_total = e0

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # if clip_fea is not None:
        #     context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        #     context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e_total,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
