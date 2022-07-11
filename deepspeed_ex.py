import torch
import kisti.deepspeed_ex as deepspeed_ex
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM, AutoTokenizer, LineByLineDataset

torch.distributed.init_process_group(backend="nccl")
deepspeed_ex.init_distributed("nccl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')

config = {
    "train_batch_size": 1,
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
            "fast_init": True
        },
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-05,
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-8
        }
    },
}

dataset = LineByLineDataset(...)
data_loader = DataLoader(dataset=dataset,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size,
                         drop_last=False)

model, optimizer, _, _ = deepspeed_ex.initialize(model=model,
                                              config_params=config,
                                              model_parameters=model.parameters())

for step, batch in enumerate(data_loader):
    input_ids, attention_masks, labels = batch

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
    loss = outputs.loss

    model.backward(loss)
    model.step()