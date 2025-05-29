import torch


class Network(torch.nn.Module):
    def __init__(self, layer_list, **kwargs):
        """
        Create a linear network with hidden layers
        """
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.final_activation = torch.nn.Sigmoid()
        self.layers = self._build_layers(layer_list)

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
