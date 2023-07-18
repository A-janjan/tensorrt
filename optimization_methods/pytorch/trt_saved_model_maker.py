from torchvision import models
import torch
from torch2trt import torch2trt


# load pretrained resnet model (for example)
model = models.resnet50(pretrained=True)


# create example data
x = torch.ones((1, 3, 224, 224)).cuda()
model = model.cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

#  saving the model
torch.save(model_trt.state_dict(), 'resnet_trt.pt')