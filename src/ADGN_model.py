import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from transformers import RobertaModel, RobertaTokenizerFast
import pickle


# E. Aspect Dependency Representation Module
class ADG(nn.Module):
    def __init__(self, args, device, num_aspects, hidden_dim):
        super().__init__()
        self.num_aspects = num_aspects
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_hops = 2

        self.aspect_polarity_fusion = nn.ModuleDict({
            "neg": nn.Linear(hidden_dim * 2, hidden_dim),
            "pos": nn.Linear(hidden_dim * 2, hidden_dim),
        })

        self.activation = nn.GELU()

        # Load raw dependency matrix
        with open(f'{args.ADG_path}/raw_dependency_matrix.pkl', 'rb') as f:
            dependency_matrix_full = pickle.load(
                f)

        dependency_matrix_full = torch.tensor(
            dependency_matrix_full, dtype=torch.float32, device=self.device)

        self.register_buffer(
            'dependency_matrix_full', dependency_matrix_full)

        self.aspect_dependency_gate = nn.Parameter(
            torch.zeros(2 * self.num_aspects, 1)
        )

    def _normalize_dependency_matrix(self, eps: float = 1e-6):

        probability_matrix = self.dependency_matrix_full.clamp(
            min=eps, max=1 - eps)

        # Self-connection clamping
        mask = torch.eye(probability_matrix.size(
            0), device=probability_matrix.device).bool()

        probability_matrix = probability_matrix.masked_fill(
            mask, eps)

        # 1) Log-odds transform
        logit_matrix = torch.log(probability_matrix) - \
            torch.log(1 - probability_matrix)

        # 2) Row-wise centering
        logit_matrix_centered = logit_matrix - \
            logit_matrix.mean(dim=1, keepdim=True)

        dependency_matrix_normalized = logit_matrix_centered

        # 3) Self-connection masking
        N2 = dependency_matrix_normalized.size(0)
        N = N2 // 2
        idx_neg = torch.arange(
            0, N2, 2, device=dependency_matrix_normalized.device)
        idx_pos = idx_neg + 1

        dependency_matrix_normalized = dependency_matrix_normalized.clone().detach()

        # Self aspect masking: neg->neg, pos->pos = 0
        dependency_matrix_normalized[idx_neg, idx_neg] = 0.0
        dependency_matrix_normalized[idx_pos, idx_pos] = 0.0

        # Self aspect masking: neg->pos, pos->neg = 0
        dependency_matrix_normalized[idx_neg, idx_pos] = 0.0
        dependency_matrix_normalized[idx_pos, idx_neg] = 0.0

        # Sort order : [a1_neg,...,aN_neg, a1_pos,...,aN_pos]
        N = self.num_aspects
        perm = torch.empty(2*N, dtype=torch.long,
                           device=dependency_matrix_normalized.device)
        for k in range(N):
            perm[k] = 2*k  # neg block
            perm[N + k] = 2*k + 1  # pos block

        dependency_matrix_normalized = dependency_matrix_normalized[perm][:, perm]

        return dependency_matrix_normalized

    def forward(self, aspect_embeddings, polarity_embeddings):

        # E.1. Construction of an Aspect Dependency Graph
        dependency_matrix_normalized = self._normalize_dependency_matrix()

        # E.2. Dependency-aware aspect representation
        # 1) Concat aspect-polarity (polarity-specific aspect representations)
        aspect_neg_embeddings = self.aspect_polarity_fusion["neg"](torch.cat(
            [aspect_embeddings, polarity_embeddings[0].expand(self.num_aspects, -1)], dim=-1))  # (N, H)

        aspect_pos_embeddings = self.aspect_polarity_fusion["pos"](
            torch.cat([aspect_embeddings, polarity_embeddings[1].expand(self.num_aspects, -1)], dim=-1))  # (N, H)

        # 2) Stack all polarity-specific aspect representations
        aspect_polarity_embeddings_initial = torch.cat(
            [aspect_neg_embeddings, aspect_pos_embeddings], dim=0
        )  # (2N, H)

        # 3) Multi-hop propagation
        propagated_embeddings_step = aspect_polarity_embeddings_initial
        multi_hop_embeddings_history = [aspect_polarity_embeddings_initial]

        for _ in range(self.num_hops):
            propagated_embeddings_step = self.activation(
                dependency_matrix_normalized @ propagated_embeddings_step
            )
            multi_hop_embeddings_history.append(
                propagated_embeddings_step)

        aspect_polarity_embeddings_propagated = sum(
            multi_hop_embeddings_history) / len(multi_hop_embeddings_history)  # (2N, H)

        # 4) Residual gating
        gate_values = torch.sigmoid(
            self.aspect_dependency_gate)  # (2N, 1)

        Mad_final = gate_values * aspect_polarity_embeddings_propagated + \
            (1 - gate_values) * aspect_polarity_embeddings_initial  # (2N, H)

        Mad_neg = Mad_final[:self.num_aspects, :]  # (N, H)
        Mad_pos = Mad_final[self.num_aspects:, :]  # (N, H)

        return Mad_pos, Mad_neg


# D. Syntactic Representation Module
class SyntacticGAT(nn.Module):
    def __init__(self, args, in_features, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.gat_out_dim = self.hidden_dim // self.n_heads
        self.dropout = args.dropout_rate
        self.gat = GATConv(in_channels=in_features,
                           out_channels=self.gat_out_dim,
                           heads=self.n_heads,
                           dropout=self.dropout,
                           concat=True)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.activation(x)
        return x


# F. Cross-Attention
class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, n_aspect):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_aspect = n_aspect

        self.Wq = nn.Linear(hidden_dim, hidden_dim)  # aspect -> query
        self.Wk = nn.Linear(hidden_dim, hidden_dim)  # token  -> key
        self.Wv = nn.Linear(hidden_dim, hidden_dim)  # token  -> value

        self.dep_fuse = nn.Linear(2*hidden_dim, hidden_dim)

    def _attend_A2T(self, A, Mx, attn_mask):

        B, L, H = Mx.shape
        Q = self.Wq(A)   # (N, H)
        K = self.Wk(Mx)  # (B, L, H)
        V = self.Wv(Mx)  # (B, L, H)

        score = torch.einsum('nh,blh->bnl', Q, K) / (H ** 0.5)   # (B, N, L)

        if attn_mask is not None:
            score = score.masked_fill(
                attn_mask.unsqueeze(1) == 0, -1e4)

        weight = torch.softmax(score, dim=-1)
        r = torch.einsum('bnl,blh->bnh', weight, V)

        return r

    def forward(self, Mx, aspect_embedding, Mad_p, Mad_n, attn_mask):
        M_base = self._attend_A2T(aspect_embedding, Mx, attn_mask)  # (B, N, H)

        M_pos = self._attend_A2T(Mad_p, Mx, attn_mask)  # (B, N, H)
        M_neg = self._attend_A2T(Mad_n, Mx, attn_mask)  # (B, N, H)
        M_dep = self.dep_fuse(torch.cat([M_pos, M_neg], dim=-1))  # (B, N, H)

        return M_base, M_dep


# C. Contextual Representation Module
def get_representation(aspects, tokenizer, model):
    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        inputs = tokenizer(aspects, return_tensors="pt",
                           padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        # Mean pooling over valid tokens
        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp_min(1e-6)
        representation_embedding = sum_hidden / lengths

    return representation_embedding


class ADGN(nn.Module):
    def __init__(self, args, device, aspects):
        super().__init__()
        self.device = device
        self.aspects = aspects
        self.n_aspect = len(self.aspects)
        self.initial_dim = 768
        self.hidden_dim = 768

        # C. Contextual Representation Module
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.contextual_model = RobertaModel.from_pretrained(
            "roberta-base").to(self.device)

        # C.2. Aspect Category Representation
        aspect_embedding = get_representation(
            aspects, self.tokenizer, self.contextual_model).to(self.device)
        self.Ma = nn.Parameter(
            aspect_embedding,  requires_grad=True)

        # C.3. Sentiment Polarity Representation
        polarity_embedding = get_representation(["It is bad", "It is great"],
                                                self.tokenizer, self.contextual_model).to(self.device)
        self.Mp = nn.Parameter(polarity_embedding, requires_grad=True)

        # D.2. Syntactic Text Representation
        self.n_heads = 4
        self.syntactic_gat1 = SyntacticGAT(
            args=args, in_features=self.initial_dim, hidden_dim=self.hidden_dim, n_heads=self.n_heads)
        self.syntactic_gat2 = SyntacticGAT(
            args=args, in_features=self.hidden_dim, hidden_dim=self.hidden_dim, n_heads=self.n_heads)

        # E. Aspect Dependency Representation Module
        self.adg = ADG(args=args, device=self.device,
                       num_aspects=self.n_aspect, hidden_dim=self.hidden_dim)

        # F. Cross-Attention
        self.cross_attention_blocks = nn.ModuleDict({
            "Mc": CrossAttention(self.hidden_dim, self.n_aspect),
            "Ms": CrossAttention(self.hidden_dim, self.n_aspect)
        })

        # G.2. Stream-wise Prediction
        self.fcn_blocks = nn.ModuleDict({
            "base": nn.Sequential(
                nn.LayerNorm(2*self.hidden_dim),
                nn.Linear(2*self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
            ),

            "dep": nn.Sequential(
                nn.LayerNorm(2*self.hidden_dim),
                nn.Linear(2*self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
            ),
        })

        # G.3. Inter-stream Fusion
        self.fusion_fcn = nn.Sequential(
            nn.Linear(4*self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, syntax_graph):

        # C.1 Text Representation
        Mc = self.contextual_model(
            input_ids, attention_mask).last_hidden_state  # (B, L, H)

        # D.2 Syntactic Text Representation
        B, L, H = Mc.shape
        edges = syntax_graph.nonzero(as_tuple=False)
        src = edges[:, 0] * L + edges[:, 1]
        dst = edges[:, 0] * L + edges[:, 2]
        batched_edge_index = torch.stack(
            [src, dst], dim=0)
        x_flatten = Mc.view(B * L, H)

        gat_output = self.syntactic_gat1(
            x_flatten, batched_edge_index)
        gat_output = self.syntactic_gat2(
            gat_output, batched_edge_index)

        Ms = gat_output.view(B, L, -1)  # (B, L, H)

        # E.2 Dependency-aware aspect representation
        M_ad_pos, M_ad_neg = self.adg(
            aspect_embeddings=self.Ma,
            polarity_embeddings=self.Mp
        )  # (N, H), (N, H)

        # F. Cross-Attention
        Mc_base, Mc_dep = self.cross_attention_blocks['Mc'](
            Mc, self.Ma, M_ad_pos, M_ad_neg, attention_mask)

        Ms_base, Ms_dep = self.cross_attention_blocks['Ms'](
            Ms, self.Ma, M_ad_pos, M_ad_neg, attention_mask)

        # G.1 Intra-stream Fusion
        M_base = torch.cat([Mc_base, Ms_base], dim=-1)  # (B, N, 2H)
        M_dep = torch.cat([Mc_dep,  Ms_dep],  dim=-1)  # (B, N, 2H)

        # G.2 Stream-wise Prediction
        y_base = self.fcn_blocks["base"](M_base).squeeze(-1)  # (B, N)
        y_dep = self.fcn_blocks["dep"](M_dep).squeeze(-1)  # (B, N)

        # G.3 Inter-stream Fusion
        gate_inp = torch.cat([M_base, M_dep], dim=-1)  # (B, N, 4H)
        beta = torch.sigmoid(self.fusion_fcn(gate_inp).squeeze(-1))  # (B, N)
        y_final = (1 - beta) * y_base + beta * y_dep  # (B, N)

        return y_final
