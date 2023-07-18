from torch2trt import TRTModule
import torch


# loading trt resnet 

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('resnet_trt.pt'))
model_trt.eval() # i'm not sure (please check)

# loading native pytorch resnet

model_native = torch.load('resnet_native.pt')
model_native.eval()


################################# Comparison #####################################

# parameters of comparison : accuracy , loss , time

################### First : providing data #######################################
########## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> we're going to use CIFAR10 dataset

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(root='cifar10',
                                   train=False,
                                   download=True,
                                   transform=transform)

# Create data loader
cifar10_loader = data.DataLoader(cifar10_dataset,
                                  batch_size=128,
                                  shuffle=False,
                                  num_workers=4)


##################################### inference ###################################

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_trt.to(device)
model_native.to(device)

################ test utils #####################

import numpy as np
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def rn50_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

# decode the results into ([predicted class, description], probability)
def predict(img_path, model):
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return d[str(ind.item())], sm_output[ind] #([predicted class, description], probability)

def benchmark(model, input_shape=(1024, 1, 224, 224), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


###################################### testing #############################################

#######  test native model
print("########################## testing native model ########################################")
benchmark(model_native, input_shape=(128, 3, 224, 224), nruns=100)
print("########################################################################################")


###### test trt modek
print("########################## testing trt model ########################################")
benchmark(model_trt, input_shape=(128, 3, 224, 224), nruns=100)
print("############################### THE END ###########################################")