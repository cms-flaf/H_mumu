import torch


class Network(torch.nn.Module):
    def __init__(self, device, layer_list, dropout=None, **kwargs):
        """
        Create a linear network with hidden layers
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        #self.activation = torch.nn.ReLU()
        self.binary_classification = layer_list[-1] == 1
        if self.binary_classification:
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = torch.nn.Softmax(dim=-1)
        self.layers = self._build_layers(layer_list)
        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if device is not None:
            self.to(device)
            self.double()

    def _build_layers(self, layer_list):
        layers = []
        ins = layer_list
        outs = layer_list[1:]
        for a, b in zip(ins, outs):
            layers.append(torch.nn.Linear(a, b))
        return torch.nn.ModuleList(layers)

    def forward(self, x):
        """
        Feed forward an input vector x.
        Different from self.predict because of BCEWithLogitsLoss:
        https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
        for layer in self.layers[:-1]:
            if self.dropout:
                x = self.dropout(x)
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        """
        Produces the final discrimination variable for an input vector x.
        """
        x = self.forward(x)
        return self.final_activation(x)