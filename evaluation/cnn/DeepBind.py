import torch.nn as nn

class DeepBind(nn.Module):

    def __init__(self):
        super(DeepBind, self).__init__()
        self.Convolutions = nn.Sequential(
            nn.ZeroPad2d((11, 12, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 24)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16)
        )
        self.GlobalMaxPool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.Dense = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, input_ids=None, labels=None):
        # input_ids expected shape: (batch, channel=1, height, width)
        x = input_ids.unsqueeze(1)  # => shape: (batch_size, 1, 4, 124)
        x = x.float()
        x = self.Convolutions(x)
        x = self.GlobalMaxPool(x)
        x = self.flatten(x)
        logits = self.Dense(x)  # shape: (batch_size, 1)
        loss_fn = nn.BCELoss()
        if labels is not None:
            labels = labels.float().view(-1, 1)  
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}