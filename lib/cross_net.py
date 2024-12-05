import torch
import torch.nn.functional as F
import math
import torch.nn as nn

from lib.xttn import mask_xattn_one_text


def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

    def forward(self, tokens, attention_x, attention_y):
        B_v, L_v, C = tokens.size()

        # (B_v, L_v)
        score = attention_x + attention_y

        num_keep_token = math.ceil(L_v * self.sparse_ratio)

        # select the top-k index, (B_v, L_v)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)

        # (B_v, L_v * token_ratio)
        keep_policy = score_index[:, :num_keep_token]

        # (B_v, L_v)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)

        # (B_v, L_v * token_ratio, C)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # fusion token
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_policy = score_index[:, num_keep_token:]

        # (B_v, L_v *  (1 - token_ratio), C )
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_score = score_sort[:, num_keep_token:]
        # through softmax function, (B_v, L_v *  (1 - token_ratio) ) -> (B_v, L_v *  (1 - token_ratio), 1)
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)

        # get fusion token (B_v, 1, C)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token, score_mask


# dim_ratio affect GPU memory
class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches)
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, keep_policy=None):
        # (B, N, C) -> (B, N, N_s)
        weight = self.weight(x)
        #  (B, N, N_s) -> (B, N_s, N)
        weight = weight.transpose(2, 1) * self.scale

        if keep_policy is not None:
            # keep_policy (B, N) -> (B, 1, N)
            keep_policy = keep_policy.unsqueeze(1)
            # increase a large number for mask patches
            weight = weight - (1 - keep_policy) * 1e10

        # learning a set of weight matrices
        weight = F.softmax(weight, dim=2)

        # (B, N_s, C)
        # multiply with patch features
        x = torch.bmm(weight, x)

        return x


class TokenAggregationforCap(nn.Module):
    def __init__(self, dim=512, dim_ratio=0.2):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)

        self.layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()

        self.scale = nn.Parameter(torch.ones(1, 1))

    def forward(self, x, keeped_tokens):
        # (N, N_m) -> (N, N_s)
        weight = self.layer_norm(x)
        weight = self.fc1(weight)
        weight = self.gelu(weight)
        fc2 = nn.Linear(weight.size(-1), keeped_tokens).to(weight.device)
        weight = fc2(weight)

        #  (N, N_s) -> (N_s, N)
        weight = weight.transpose(1, 0) * self.scale

        # learning a set of weight matrices
        weight = F.softmax(weight, dim=1)

        # (N_s, C)
        # multiply with token features
        x = torch.matmul(weight, x)

        return x


## sparse + aggregation
class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.opt = opt

        self.hidden_dim = opt.embed_size
        self.num_patches = opt.num_patches

        self.img_sparse_ratio = opt.img_sparse_ratio
        self.img_aggr_ratio = opt.img_aggr_ratio

        self.cap_aggr_ratio = opt.cap_aggr_ratio

        self.attention_weight = opt.attention_weight
        self.monitor_weight = opt.monitor_weight

        # the number of aggregated patches
        self.keeped_patches = int(self.num_patches * self.img_aggr_ratio * self.img_sparse_ratio)
        # sparse network
        self.img_sparse_net = TokenSparse(embed_dim=self.hidden_dim,
                                          sparse_ratio=self.img_sparse_ratio,
                                          )

        # aggregation network
        self.img_aggr_net = TokenAggregation(dim=self.hidden_dim,
                                             keeped_patches=self.keeped_patches,
                                             )
        self.cap_aggr_net = TokenAggregationforCap(dim=self.hidden_dim)

    def forward(self, img_embs, cap_embs, cap_lens):

        B_v, L_v, C = img_embs.shape

        # feature normalization
        # (B_v, L_v, C)
        img_embs_norm = F.normalize(img_embs, dim=-1)
        # (B_t, L_t, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)

        self.img_has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        #  whether it exists [cls] token
        if self.img_has_cls_token:
            # (B_v, 1, C)
            img_cls_emb = img_embs[:, 0:1, :]
            img_cls_emb_norm = img_embs_norm[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]

        else:
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # compute self-attention
        with torch.no_grad():
            # (B_v, L_v, C) ->  (B_v, 1, C)
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            # (B_v, L_v, C) -> (B_v, L_v)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        score_img_mask_all = []
        min_keep_tokens = 4
        max_keep_tokens = 100
        for i in range(len(cap_lens)):

            n_word = cap_lens[i]

            # (N_m, C)
            cap_i = cap_embs[i, :n_word, :]

            ## compute cross-attention
            with torch.no_grad():
                # (N_m, C) -> (1, C) -> (1, 1, C)
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                # (B_v, L_v, C) -> (B_v, L_v)
                img_spatial_cap_i_attention = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # selection
            select_img_tokens, extra_img_token, score_img_mask = self.img_sparse_net(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,
                attention_y=img_spatial_cap_i_attention,
            )
            # aggregation
            aggr_img_tokens = self.img_aggr_net(select_img_tokens)

            # aggr_tokens = select_tokens

            # add fusion token
            keep_img_spatial_tokens = torch.cat([aggr_img_tokens, extra_img_token], dim=1)

            # add [cls] token
            if self.img_has_cls_token:
                select_img_tokens = torch.cat((img_cls_emb, keep_img_spatial_tokens), dim=1)
            else:
                select_img_tokens = keep_img_spatial_tokens

            # patch normalization
            select_img_tokens = F.normalize(select_img_tokens, dim=-1)

            # Text
            # (N_m - 1, C)
            cap_i_spatial = cap_i[1:, :]

            cap_i_cls = cap_i[0:1, :]

            keep_tokens = max(min_keep_tokens, min(int(self.cap_aggr_ratio * cap_i_spatial.size(0)), max_keep_tokens))
            # (N_s, C)
            aggr_cap_tokens = self.cap_aggr_net(cap_i_spatial, keep_tokens)
            # (N_s + 1, C)
            select_cap_tokens = torch.cat((cap_i_cls, aggr_cap_tokens), dim=0)

            select_cap_tokens = F.normalize(select_cap_tokens, dim=-1)
            # (B_v, N_s + 1, C)
            select_cap_tokens = select_cap_tokens.repeat(B_v, 1, 1)

            # image-text similarity
            # (B_v, 1)
            sim_one_text = mask_xattn_one_text(img_embs=select_img_tokens,
                                               cap_embs=select_cap_tokens,
                                               )

            improve_sims.append(sim_one_text)
            score_img_mask_all.append(score_img_mask)
        # Introduce text information
        # process each text separately

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)
        score_img_mask_all = torch.stack(score_img_mask_all, dim=0)

        if self.training:
            return improve_sims, score_img_mask_all
        else:
            return improve_sims


if __name__ == '__main__':
    pass
