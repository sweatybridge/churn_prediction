import json

import torch
from torchvision.models import resnet50
# from torchvision.models import resnet152


def main():
    # model = resnet152(pretrained=True)
    # torch.save(model, "/artefact/model.pth")
    small = resnet50(pretrained=True)
    torch.save(small, "/artefact/small.pth")
    with open("/artefact/data.json", "w") as f:
        json.dump({i: i ** 2 for i in range(100)}, f)


if __name__ == "__main__":
    main()
