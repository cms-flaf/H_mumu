
import torch


class ExportNetwork(torch.nn.Module):
    def __init__(self, model, device, renorm_mean, renorm_std):
        """
        Wrapper to include input renorm as an initial layer
        """
        super().__init__()
        self.device = device
        self.model = model
        if device is not None:
            self.to(device)
            self.double()
            if renorm_mean is not None and renorm_std is not None:
                self.mean = torch.tensor(renorm_mean, device=self.device, dtype=torch.double)
                self.std = torch.tensor(renorm_std, device=self.device, dtype=torch.double)
            else:
                self.mean = None
                self.std = None
        else:
            if renorm_mean is not None and renorm_std is not None:
                self.mean = torch.tensor(renorm_mean, device=self.device)
                self.std = torch.tensor(renorm_std, device=self.device)
            else:
                self.mean = None
                self.std = None


    def forward(self, x):
        if self.mean is not None and self.std is not None:
            x = x.sub(self.mean).div(self.std)
        return self.model(x)

def export_to_onnx(x_data, model, outname, device, renorm_mean, renorm_std):
    if device is None:
        x = torch.tensor(x_data, device=device)
    else:
        x = torch.tensor(x_data, device=device, dtype=torch.double)

    export_model = ExportNetwork(model, device, renorm_mean, renorm_std)

    torch.onnx.export(
        model=export_model,
        args=(x[0:3]),
        f=outname + ".onnx",
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": [0], "y": [0]},
    )