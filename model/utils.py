from model.resnet_cifar import resnet32
from model.resnet import resnet50
import torch

def create_model(num_classes, gpu, use_norm=False):
    model = resnet32(num_classes, use_norm)

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    return model
