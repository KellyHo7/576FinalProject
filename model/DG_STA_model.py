import torch
import torch.nn as nn
from .attention_layer import *



class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate):
        super(DG_STA, self).__init__()

        head_num = 8 # head number of the spatial and temporal attention models
        head_dim = 32 # dimension of the query, key, and value vector
        in_features = 3
        normalized_shape = 128

        self.l1 = nn.Sequential(
            nn.Linear(in_features, normalized_shape), #(3, 128)
            nn.ReLU(),
            nn.LayerNorm(normalized_shape),
            nn.Dropout(dp_rate)
        )

        self.satt = ST_ATT_Layer(input_size=128,output_size=128,head_num=head_num,head_dim=head_dim,dp_rate=dp_rate,time_len=8,att_type="spatial")

        self.l2 = nn.Sequential(
            nn.Linear(head_num * head_dim, normalized_shape), #(256, 128)
            nn.ReLU(),
            nn.LayerNorm(normalized_shape),
            nn.Dropout(dp_rate)
        )
        
        self.tatt = ST_ATT_Layer(input_size=128,output_size=128,head_num=head_num,head_dim=head_dim,dp_rate=dp_rate,time_len=8,att_type="temporal")

        self.l3 = nn.Sequential(
            nn.Linear(head_num * head_dim, normalized_shape), #(256, 128)
            nn.ReLU(),
            nn.LayerNorm(normalized_shape),
            nn.AvgPool2d(1),
            nn.Dropout(dp_rate)
        )

        self.fc = nn.Linear(normalized_shape, num_classes)



    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1] 
        joint_num = x.shape[2] 

        #reshape x
        x = x.reshape(-1,time_len * joint_num,3)

        #first layer
        x = self.l1(x)
        
        #spatal
        x = self.satt(x)
        
        #second layer
        x = self.l2(x)
        
        #temporal
        x = self.tatt(x)
        
        #third layer
        x = self.l3(x)
        
        #fc layer
        x = x.sum(1) / x.shape[1] 
        x = self.fc(x)

        return x
