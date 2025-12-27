import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from wan.modules.model import WanModel, sinusoidal_embedding_1d

class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, action):
        """
        Args:
            action: (B, T, action_dim)
        Returns:
            (B, T, output_dim)
        """
        return self.mlp(action)

class WanActionModel(nn.Module):
    def __init__(self, base_model, action_dim, hidden_dim=1024, lora_rank=16):
        super().__init__()
        self.base_model = base_model
        # Action encoder outputs embeddings of the same dimension as the transformer
        self.action_encoder = ActionEncoder(action_dim, hidden_dim, base_model.dim)
        
        # Projection for AdaLN parameters (6 parameters: shift/scale for self-attn, cross-attn, and ffn)
        self.action_adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(base_model.dim, base_model.dim * 6)
        )

        # Ensure new modules are in the same dtype as the base model
        dtype = base_model.patch_embedding.weight.dtype
        self.action_encoder.to(dtype)
        self.action_adaln_proj.to(dtype)

        # Apply LoRA to the base model
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
            lora_dropout=0.05,
            bias="none",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
    def forward(self, x, t, context, seq_len, actions, clip_fea=None, y=None, **kwargs):
        """
        Args:
            x: Batch tensor (B, C, F, H, W) or List of input video tensors [C, F, H, W]
            t: Timesteps (B,)
            context: Batch tensor (B, L, D) or List of text embeddings [L, D]
            actions: (B, T, action_dim)
            clip_fea: (B, 257, 1280) CLIP features
            y: Batch tensor (B, C, F, H, W) or List of conditional video tensors [C, F, H, W]
        """
        # 1. Encode actions
        # Ensure actions are in the same dtype as action_encoder
        dtype = self.action_encoder.mlp[0].weight.dtype
        action_embeds = self.action_encoder(actions.to(dtype)) # (B, T, D)
        
        # 2. Prepare base model inputs
        device = self.base_model.patch_embedding.weight.device
        
        # Convert batch tensor to list format if needed (for compatibility)
        is_batch_tensor = isinstance(x, torch.Tensor) and x.ndim == 5
        if is_batch_tensor:
            x = [x[i] for i in range(x.shape[0])]
            if y is not None:
                y = [y[i] for i in range(y.shape[0])]
            if isinstance(context, torch.Tensor) and context.ndim == 3:
                context = [context[i] for i in range(context.shape[0])]
        
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Embeddings
        x = [self.base_model.patch_embedding(u.unsqueeze(0).to(device)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        
        # Pad x to max seq_len
        max_seq_len = max(seq_len, seq_lens.max().item())
        x_padded = torch.cat([
            torch.cat([u, u.new_zeros(1, max_seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # 3. Time embeddings
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e_time = self.base_model.time_embedding(
                sinusoidal_embedding_1d(self.base_model.freq_dim, t).float().to(device))
            e0_time = self.base_model.time_projection(e_time).unflatten(1, (6, self.base_model.dim)) # (B, 6, D)
            
        # 4. Action AdaLN (Map T frames to L tokens)
        # Expand action_embeds (B, T, D) to (B, L, D)
        action_embeds_expanded = []
        for i in range(len(x)):
            f, h, w = grid_sizes[i].tolist()
            # Repeat each frame's action embedding for all patches in that frame
            # action_embeds[i] is (T, D), we take first f frames
            t_embeds = action_embeds[i, :f] # (f, D)
            s_embeds = t_embeds.repeat_interleave(h * w, dim=0) # (f*h*w, D)
            # Pad to max_seq_len
            padding = s_embeds.new_zeros(max_seq_len - s_embeds.size(0), s_embeds.size(1))
            action_embeds_expanded.append(torch.cat([s_embeds, padding], dim=0))
        
        action_embeds_expanded = torch.stack(action_embeds_expanded) # (B, L, D)
        
        # Project to AdaLN parameters
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # Ensure action_embeds_expanded is in the same dtype as action_adaln_proj
            proj_dtype = self.action_adaln_proj[1].weight.dtype
            e0_action = self.action_adaln_proj(action_embeds_expanded.to(proj_dtype)) # (B, L, 6*D)
            e0_action = e0_action.unflatten(2, (6, self.base_model.dim)).permute(0, 2, 1, 3) # (B, 6, L, D)
            
            # Combine Time and Action modulation
            # e0_time is (B, 6, D), we unsqueeze to (B, 6, 1, D) for broadcasting
            combined_e0 = e0_time.unsqueeze(2) + e0_action # (B, 6, L, D)

        # 5. Text Context
        context_tensors = self.base_model.text_embedding(
            torch.stack([
                torch.cat([u.to(device), u.new_zeros(self.base_model.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.base_model.img_emb(clip_fea)
            context_tensors = torch.concat([context_clip, context_tensors], dim=1)

        # 6. Run Blocks
        kwargs_blocks = dict(
            e=combined_e0, # Now per-token modulation!
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.base_model.freqs.to(device),
            context=context_tensors,
            context_lens=None)

        for block in self.base_model.blocks:
            x_padded = block(x_padded, **kwargs_blocks)

        # 7. Head (using time embedding for global modulation)
        x_padded = self.base_model.head(x_padded, e_time)

        # 8. Unpatchify
        out = self.base_model.unpatchify(x_padded, grid_sizes)
        return [u.float() for u in out]

    def save_lora_weights(self, path):
        state_dict = get_peft_model_state_dict(self.base_model)
        action_encoder_dict = self.action_encoder.state_dict()
        action_adaln_dict = self.action_adaln_proj.state_dict()
        torch.save({
            "lora": state_dict,
            "action_encoder": action_encoder_dict,
            "action_adaln": action_adaln_dict
        }, path)

    def load_lora_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.base_model.load_state_dict(checkpoint["lora"], strict=False)
        self.action_encoder.load_state_dict(checkpoint["action_encoder"])
        self.action_adaln_proj.load_state_dict(checkpoint["action_adaln"])
