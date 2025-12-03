import torch


def get_torch_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        # else "mps" if torch.backends.mps.is_available() else "cpu"
        # ^ You can toggle between the lines above. For me, mps is slower than cpu
    )

    return device
