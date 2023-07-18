from torchvision import models
import torch


# load pretrained resnet model (for example)
model = models.resnet50(pretrained=True)


# save model in format of pth files

### Save the entire model
torch.save(model, 'resnet_native.pt')

### Save only the model parameters
#torch.save(model.state_dict(), 'resnet_native.pt')

