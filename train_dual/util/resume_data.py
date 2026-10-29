"""Resume epoch-boundary sample order without an extra sampler RNG draw."""
import torch
from torch.utils.data import DataLoader,RandomSampler


def boundary_loader(dataset,sampler_state,*,workers=2,batch_size=1,seed=42):
    # In the original persistent loader, subsequent epochs reuse worker base
    # seeds. A freshly constructed loader would otherwise consume an additional
    # draw from the SAME RNG before sampling, changing the permutation.
    sampler_rng=torch.Generator();sampler_rng.set_state(sampler_state.cpu())
    worker_rng=torch.Generator().manual_seed(seed)
    sampler=RandomSampler(dataset,generator=sampler_rng)
    loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=workers,
        persistent_workers=workers>0,pin_memory=True,generator=worker_rng)
    return loader,sampler_rng
