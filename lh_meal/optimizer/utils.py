from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
from transformers import AdamW
from .optimizer import Ralamb
from .scheduler import WarmupLinearSchedule

def optimizer_select(model, args):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.w_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=args.lr, eps=1e-8)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    elif args.optimizer == 'Ralamb':
        optimizer = Ralamb(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr)
    else:
        raise Exception("Choose optimizer in ['AdamW', 'Adam', 'SGD', 'Ralamb']")
    return optimizer

def shceduler_select(optimizer, dataloader_dict, args):
    if args.scheduler == 'constant':
        scheduler = StepLR(optimizer, step_size=len(dataloader_dict['train']), gamma=1)
    elif args.scheduler == 'warmup':
        scheduler = WarmupLinearSchedule(optimizer, 
                                        warmup_steps=int(len(dataloader_dict['train'])*args.n_warmup_epochs), 
                                        t_total=len(dataloader_dict['train'])*args.num_epochs)
    elif args.scheduler == 'reduce_train':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(len(dataloader_dict['train'])*1.5),
                                      factor=0.5)
    elif args.scheduler == 'reduce_valid':
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    elif args.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: args.lr_lambda ** epoch)
    else:
        raise Exception("Choose shceduler in ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']")
    return scheduler