import torch
from torchvision.models import resnet152


def main():
    model = resnet152(pretrained=True)
    torch.save(model, "/artefact/model.pth")


if __name__ == "__main__":
    main()
