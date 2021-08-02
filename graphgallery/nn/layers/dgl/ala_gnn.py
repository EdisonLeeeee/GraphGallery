import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

class GatedLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes,
        bias=True,
        activation=None,
        share_tau=True,
        dropout=0.0,
        lidx=1,
    ):

        super().__init__()
        self.activation = activation
        self.share_tau = share_tau
        self.dropout = nn.Dropout(dropout)
        self.tau1 = nn.Parameter(torch.zeros((1,)))
        self.tau2 = nn.Parameter(torch.zeros((1,)))

        self.ln1 = nn.LayerNorm(num_nodes, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(num_nodes, elementwise_affine=False)
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.lidx = lidx
        self.reset_parameters(lidx)

    def reset_parameters(self, lidx=None, how="layerwise"):
        lidx = lidx or self.lidx
        # initialize params
        if how == "normal":
            nn.init.normal_(self.tau1)
            nn.init.normal_(self.tau2)
        else:
            nn.init.constant_(self.tau1, 1 / (lidx + 1))
            nn.init.constant_(self.tau2, 1 / (lidx + 1))
        return

    def forward(self, g, feats, logits, old_z, tau1=None, tau2=None):
        with g.local_scope():
            h = self.dropout(feats)

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)

            g.ndata['h'] = h
            g.ndata['logits'] = logits
            g.ndata['degree'] = degs

            g.update_all(
                message_func=fn.copy_u("logits", "logits"),
                reduce_func=adaptive_reduce_func,
            )
            f1 = g.ndata.pop("f1")
            f2 = g.ndata.pop("f2")

            norm_f1 = self.ln1(f1)
            norm_f2 = self.ln2(f2)

            if self.share_tau:
                z = torch.sigmoid((-1) * (norm_f1 - tau1)) * torch.sigmoid(
                    (-1) * (norm_f2 - tau2)
                )
            else:
                # tau for each layer
                z = torch.sigmoid((-1) * (norm_f1 - self.tau1)) * torch.sigmoid(
                    (-1) * (norm_f2 - self.tau2)
                )

            gate = torch.min(old_z, z)
            g.update_all(
                message_func=fn.copy_u("h", "feat"),
                reduce_func=fn.sum(msg="feat", out="agg"),
            )

            agg = g.ndata.pop("agg")

            normagg = agg * norm  # normalization by tgt degree

            if self.activation is not None:
                normagg = self.activation(normagg)
            new_h = h + gate.unsqueeze(1) * normagg
            return new_h, z

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, num_nodes={self.num_nodes}, share_tau={self.share_tau})"


class GatedAttnLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes,
        num_heads,
        bias=True,
        activation=None,
        share_tau=True,
        dropout=0.0,
        lidx=1,
    ):

        super().__init__()
        self.activation = activation
        self.share_tau = share_tau
        self.dropout = nn.Dropout(dropout)
        self.tau1 = nn.Parameter(torch.zeros((1,)))
        self.tau2 = nn.Parameter(torch.zeros((1,)))

        if in_features != out_features:
            self.fc = nn.Linear(
                in_features, out_features * num_heads, bias=False
            )  # for first layer
        else:
            self.fc = None

        self.ln1 = nn.LayerNorm((num_nodes, num_heads), elementwise_affine=False)
        self.ln2 = nn.LayerNorm((num_nodes, num_heads), elementwise_affine=False)
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.lidx = lidx
        self.reset_parameters(lidx)

    def reset_parameters(self, lidx=None, how="layerwise"):
        lidx = lidx or self.lidx
        if how == "normal":
            nn.init.normal_(self.tau1)
            nn.init.normal_(self.tau2)
        else:
            nn.init.constant_(self.tau1, 1 / (lidx + 1))
            nn.init.constant_(self.tau2, 1 / (lidx + 1))

        return

    def forward(
        self,
        g,
        h,
        logits,
        old_z,
        attn_l,
        attn_r,
        *,
        shared_tau=True,
        tau1=None,
        tau2=None
    ):
        with g.local_scope():
            h = self.dropout(h)

            if self.fc is not None:
                feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
            else:
                feat = h
            g.ndata["h"] = feat  # (n_node, n_feat)
            g.ndata["logits"] = logits

            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feat.device).unsqueeze(1)
            g.ndata["degree"] = degs

            el = (feat * attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat * attn_r).sum(dim=-1).unsqueeze(-1)
            g.ndata.update({"ft": feat, "el": el, "er": er})
            # compute edge attention
            g.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(g.edata.pop("e"))
            # compute softmax
            g.edata["a"] = self.dropout(edge_softmax(g, e))

            g.update_all(
                message_func=adaptive_attn_message_func,
                reduce_func=adaptive_attn_reduce_func,
            )
            f1 = g.ndata.pop("f1")
            f2 = g.ndata.pop("f2")
            norm_f1 = self.ln1(f1)
            norm_f2 = self.ln2(f2)
            if shared_tau:
                z = torch.sigmoid((-1) * (norm_f1 - tau1)) * torch.sigmoid(
                    (-1) * (norm_f2 - tau2)
                )
            else:
                # tau for each layer
                z = torch.sigmoid((-1) * (norm_f1 - self.tau1)) * torch.sigmoid(
                    (-1) * (norm_f2 - self.tau2)
                )

            gate = torch.min(old_z, z)

            agg = g.ndata.pop("agg")
            normagg = agg * norm.unsqueeze(1)  # normalization by tgt degree

            if self.activation:
                normagg = self.activation(normagg)
            new_h = feat + gate.unsqueeze(2) * normagg
            return new_h, z
        
def adaptive_reduce_func(nodes):
    """
    compute metrics and determine if we need to do neighborhood aggregation.
    """
    # (n_nodes, n_edges, out_features)
    _, pred = torch.max(nodes.mailbox["logits"], dim=2)
    _, center_pred = torch.max(nodes.data["logits"], dim=1)
    n_degree = nodes.data["degree"]
    device = n_degree.device
    # case 1
    # ratio of common predictions
    f1 = torch.sum(torch.eq(pred, center_pred.unsqueeze(1)), dim=1) / n_degree
    f1 = f1.detach()
    # case 2
    # entropy of neighborhood predictions
    uniq = torch.unique(pred)
    # (n_unique)
    cnts_p = torch.zeros(
        (
            pred.size(0),
            uniq.size(0),
        )
    )
    for i, val in enumerate(uniq):
        tmp = torch.sum(torch.eq(pred, val), dim=1) / n_degree
        cnts_p[:, i] = tmp
    cnts_p = torch.clamp(cnts_p, min=1e-5)

    f2 = (-1) * torch.sum(cnts_p * torch.log(cnts_p), dim=1)
    f2 = f2.detach()
    return {
        "f1": f1.to(device),
        "f2": f2.to(device),
    }

def adaptive_attn_message_func(edges):
    return {
        "feat": edges.src["ft"] * edges.data["a"],
        "logits": edges.src["logits"],
        "a": edges.data["a"],
    }


def adaptive_attn_reduce_func(nodes):
    _, pred = torch.max(nodes.mailbox["logits"], dim=2)
    _, center_pred = torch.max(nodes.data["logits"], dim=1)
    n_degree = nodes.data["degree"]
    device = n_degree.device
    # case 1
    # ratio of common predictions
    a = nodes.mailbox["a"].squeeze(3)  # (n_node, n_neighbor, n_head, 1)
    n_head = a.size(2)
    idxs = torch.eq(pred, center_pred.unsqueeze(1)).unsqueeze(2).expand_as(a)
    f1 = torch.div(
        torch.sum(a * idxs, dim=1), n_degree.unsqueeze(1)
    )  # (n_node, n_head)
    f1 = f1.detach()
    # case 2
    # entropy of neighborhood predictions
    uniq = torch.unique(pred)
    # (n_unique)
    cnts_p = torch.zeros(
        (
            pred.size(0),
            n_head,
            uniq.size(0),
        )
    ).cuda()
    for i, val in enumerate(uniq):
        idxs = torch.eq(pred, val).unsqueeze(2).expand_as(a)
        tmp = torch.div(
            torch.sum(a * idxs, dim=1), n_degree.unsqueeze(1)
        )  # (n_nodes, n_head)
        cnts_p[:, :, i] = tmp
    cnts_p = torch.clamp(cnts_p, min=1e-5)
    f2 = (-1) * torch.sum(cnts_p * torch.log(cnts_p), dim=2)
    f2 = f2.detach()
    neighbor_agg = torch.sum(nodes.mailbox["feat"], dim=1)  # (n_node, n_head, n_feat)
    return {
        "f1": f1.to(device),
        "f2": f2.to(device),
        "agg": neighbor_agg.to(device),
    }