import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
import torch


class ActorCritic(nn.Module):
    def __init__(
        self,
        # feature extraction net unique attributes:
        num_layers,
        learn_eps,
        neighbor_pooling_type,
        input_dim,
        hidden_dim,
        # feature extraction net MLP attributes:
        num_mlp_layers_feature_extract,
        # actor net MLP attributes:
        num_mlp_layers_actor,
        hidden_dim_actor,
        # critic net MLP attributes:
        num_mlp_layers_critic,
        hidden_dim_critic,
        init_method,
        # actor/critic/feature_extraction shared attribute
        device,
    ):
        super(ActorCritic, self).__init__()
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1, init_method).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1, init_method).to(device)

    def forward(
        self,
        x,
        n_j,
        graph_pool,
        padded_nei,
        adj,
        candidate,
        mask,
    ):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        # prepare policy feature: concat omega feature with global feature
        dummy = candidate.unsqueeze(-1).expand(-1, n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        '''# prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_j, self.n_m))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(omega, torch.ones_like(omega, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob'''

        # concatenate feature
        # concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        return pi, v


if __name__ == '__main__':
    print('Go home')