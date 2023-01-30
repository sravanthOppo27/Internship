from torchvision import models
import torch
import torch.quantization
import os
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
from resnet import resnet18
import time
cpu_device = torch.device("cpu:0")
gpu_device =  torch.device("cuda:0")


class QuantizedResNet18(torch.nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


modeldir = "/home/notebook/data/group/ILSVRC/Data/CLS-LOC/val"


def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageFolder(
        data_path, transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageFolder(
        data_path, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler)

    return data_loader, data_loader_test

train_loader , test_loader = prepare_data_loaders(modeldir)

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    tmp =  os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    
    return tmp

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    count = 0
    
    print(len(test_loader))

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0
        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        print("batch num is : ",count)
        if (count%100==0):
            print("loss for batch ",count,"is : ",loss)
        count+=1

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

model = resnet18(pretrained=True)

model.to(cpu_device)

print(type(model))

model_fused = copy.deepcopy(model)

model.eval()

model_fused.eval()

model_fused = torch.quantization.fuse_modules(model_fused,['conv1','bn1'])

qmodel = QuantizedResNet18(model_fused)

model.eval()

qmodel.eval()

qmodel.qconfig = torch.quantization.get_default_qconfig('fbgemm')

print(type(qmodel))

qmodel = torch.quantization.prepare(qmodel,inplace=True)

calibrate_model(qmodel,train_loader,device = cpu_device)

torch.quantization.convert(qmodel,inplace = True)

qmodel.eval()

qmodel.to(cpu_device)

model_size = size_of_model(model)

qmodel_size = size_of_model(qmodel)

print("Unquantized Model size in MB:",model_size)

print("Quantized Model size in MB: ",qmodel_size)

fp32_cpu_inference_latency = measure_inference_latency(model=model, device=cpu_device, input_size=(1,3,32,32),num_samples=100)

int8_cpu_inference_latency = measure_inference_latency(model=qmodel, device=cpu_device, input_size=(1,3,32,32), num_samples=100)

print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))

print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))

_,unquantized_model_accuracy = evaluate_model(model,train_loader,device=cpu_device,criterion=torch.nn.CrossEntropyLoss())

print("Unquantized model accuracy is {:.5f}".format(unquantized_model_accuracy))

_,quantized_model_accuracy = evaluate_model(qmodel,train_loader,device=cpu_device)

print("Quantized Model accuracy is {:.5f} ".format(quantized_model_accuracy))
