import torch
import torch.nn as nn
import copy

class MathTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, max_dims, embed_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.data_proj = nn.Sequential(
            nn.Linear(max_dims + 1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.data_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward_jepa(self, masked_tokens, unmasked_tokens):
        B, seq_len = masked_tokens.shape
        pos = torch.arange(seq_len, device=masked_tokens.device).unsqueeze(0).expand(B, seq_len)
        
        cx_emb = self.token_emb(masked_tokens) + self.pos_emb(pos)
        cx_latents = self.context_encoder(cx_emb)
        
        pred_latents = self.predictor(cx_latents)
        
        with torch.no_grad():
            tgt_emb = self.token_emb(unmasked_tokens) + self.pos_emb(pos)
            target_latents = self.target_encoder(tgt_emb)
            
        return pred_latents, target_latents

    def forward_autoregressive(self, X, y, tgt_seq):
        B, num_samples, dims = X.shape
        
        mean_X = X.mean(dim=1, keepdim=True)
        std_X = X.std(dim=1, keepdim=True).clamp(min=1e-5)
        norm_X = (X - mean_X) / std_X

        mean_y = y.mean(dim=1, keepdim=True)
        std_y = y.std(dim=1, keepdim=True).clamp(min=1e-5)
        norm_y = (y - mean_y) / std_y
        
        numeric_cat = torch.cat([norm_X, norm_y], dim=-1)
        
        data_emb = self.data_proj(numeric_cat)
        memory = self.data_encoder(data_emb)
        
        seq_len = tgt_seq.shape[1]
        pos = torch.arange(seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        
        tgt_emb = self.token_emb(tgt_seq) + self.pos_emb(pos)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_seq.device)
        
        decoder_out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        logits = self.head(decoder_out)
        
        return logits
        
    @torch.no_grad()
    def update_teacher(self, ema_decay=0.996):
        for s_param, t_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            t_param.data.mul_(ema_decay).add_(s_param.data, alpha=1.0 - ema_decay)
