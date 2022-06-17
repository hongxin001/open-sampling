
from alg.standard import Standard
from alg.ood_noise import OODNoise

def create_alg(args, gpu, num_classes, cls_num_list, train_dataset):
    if args.alg == "standard":
        alg = Standard(args, gpu, num_classes, cls_num_list, train_dataset)
    elif args.alg == "open":
        alg = OODNoise(args, gpu, num_classes, cls_num_list, train_dataset)

    return alg