import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class PositionalEncoding(nn.Module):
    def __init__(self, ft_size, time_len, att_type):
        super(PositionalEncoding, self).__init__()
        
        self.N = 22 # hand joint number
        self.T = time_len # time frames
        self.att_type = att_type

        pos = []
        # spatial/temporal positional embedding
        if att_type == "spatial" or att_type == "mask_s":
            for t in range(self.T):
                for joint in range(self.N):
                    pos.append(joint) # N for a hand joint
        elif att_type == "temporal" or att_type == "mask_t": 
            pos = list(range(self.N * self.T)) # NxT for a node

        # torch.from_numpy: Creates a Tensor from a numpy.ndarray.
        # torch.unsqueeze
        # Returns a new tensor with a dimension of size one inserted at the specified position.
        position = torch.from_numpy(np.array(pos)).unsqueeze(1).float()

        # torch.zeros:
        #Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
        position_embedding = torch.zeros(self.N * self.T, ft_size)
        division_term = torch.exp(torch.arange(0, ft_size, 2).float() * (-math.log(10000.0)) / ft_size)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        position_embedding[:, 0::2] = torch.sin(position * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        position_embedding[:, 1::2] = torch.cos(position * division_term)

        # Saving buffer (same as parameter without gradients needed)
        position_embedding = position_embedding.unsqueeze(0) #.cuda()
        self.register_buffer('pos_embedding', position_embedding)


    def forward(self, x):
        # Residual connection + pos embedding
        x = x + self.pos_embedding[:, :x.size(1)]
        return x
    


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, head_dim, input_dim, dp_rate, att_type):
        super(MultiHeadedAttention, self).__init__()

        self.head_num = head_num
        self.head_dim = head_dim # = d_k
        self.d_model = head_num * head_dim
        self.att_type = att_type


        self.key = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.Dropout(dp_rate))

        self.query = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.Dropout(dp_rate))

        self.value = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.Dropout(dp_rate))

        self.register_buffer('t_mask', self.masks()[0])
        self.register_buffer('s_mask', self.masks()[1])
        

    # Make temporal and spatial masks. Masks are of shape 
    # (HandSet.time_len * Number of joints, HandSet.time_len * Number of joints) to represent a graph of 
    # the handjoints in the current and future timeframes
    # - spatial mask has 1 if an element represents the spatial or self-connected edges and 0 otherwise
    # - temporal mask has 1 if an element represents the temporal or self-connected edges and 0 otherwise
    def masks(self):
        num_frames = 8
        num_joints = 22

        temporal_mask = torch.ones(num_joints * num_frames, num_joints * num_frames)
        # Set all elements in current frame to 0, except the diagonal.
        # - Diagonal elements represent self-connected edges.
        # - All elements in the current frame are spatial edges.
        for i in range(num_frames):
            temporal_mask[i * num_joints: (i * num_joints) + num_joints, \
                          i * num_joints: (i * num_joints) + num_joints] = torch.zeros(num_joints, num_joints)
        temporal_mask += torch.eye(num_joints * num_frames)

        spatial_mask = torch.zeros(num_joints * num_frames, num_joints * num_frames)
        # Set all elements in current frame to 1.
        # - Diagonal elements represent self-connected edges.
        # - All elements in the current frame are spatial edges.
        for i in range(num_frames):
            spatial_mask[i * num_joints: (i * num_joints) + num_joints, \
                          i * num_joints: (i * num_joints) + num_joints] += torch.ones(num_joints, num_joints)
        return temporal_mask, spatial_mask


    def scaled_dot_product_attention(self,query, key, value):
        dk = query.size()[-1]
        
        weights = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)

        # Section 3.4 Efficient Implementation
        if self.att_type == "temporal":
            weights *= self.t_mask  
            weights += (1 - self.t_mask) * (-9e15)  
        elif self.att_type == "spatial":
            weights *= self.s_mask
            weights += (1 - self.s_mask) * (-9e15)
        
        attention = F.softmax(weights, dim=-1)
        values = torch.matmul(attention, value)
        return values, attention


    def forward(self, x):
        bs = x.size(0)
        
        # Perform linear projections in batch from d_model = head_num x head_dim (split into h heads)
        query = self.query(x).view(bs, -1, self.head_num, self.head_dim)
        key = self.key(x).view(bs, -1, self.head_num, self.head_dim)
        value = self.value(x).view(bs, -1, self.head_num, self.head_dim)

        # Transpose to get dimensions bs * head_num * head_dim * seq_len
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # Calculate attention
        x, self.attention = self.scaled_dot_product_attention(query, key, value)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        return x



class ST_ATT_Layer(nn.Module):
    def __init__(self, input_size, output_size, head_num, head_dim, dp_rate, time_len, att_type):
        super(ST_ATT_Layer, self).__init__()

        self.PE = PositionalEncoding(input_size, time_len, att_type)
        self.ATTN = MultiHeadedAttention(head_num, head_dim, input_size, dp_rate, att_type) #do att on input dim


    def forward(self, x):
        x = self.PE(x)
        x = self.ATTN(x)
        return x
