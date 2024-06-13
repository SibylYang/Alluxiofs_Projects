import torch
from torch.utils.data import Sampler, DataLoader, Dataset

class RangeSampler(Sampler):
    def __init__(self, start_index, end_index, shuffle=False):
        self.start_index = start_index
        self.end_index = end_index
        self.shuffle = shuffle
        # self.indices = list(range(start_index, end_index))

    def __iter__(self):
        # if self.shuffle:
        #     return iter(torch.randperm(len(self.indices)).tolist())
        # else:
        #     return iter(self.indices)
        indices = []
        for _ in range(4):
            indices.extend(list(range(self.start_index, self.end_index)))
        print(indices)
        return iter(indices)


    def __len__(self):
        return len(self.indices)

# Example usage with a dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.access_num = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.access_num += 1
        return idx

# Define the dataset and the custom sampler
dataset = DummyDataset(size=100)
print(dataset.__len__())
sampler = RangeSampler(start_index=0, end_index=20, shuffle=True)

# Create DataLoader with the custom sampler
data_loader = DataLoader(dataset, sampler=sampler, batch_size=5)

# Iterate through the DataLoader
for batch in enumerate(data_loader):
    print(batch)
# loop = tqdm(enumerate(data_loader), leave=False, total=len(data_loader))
#
# # Iterate over the DataLoader with the progress bar
# for batch_idx, (data, labels) in loop:
#     # Optionally, set a description for the progress bar
#     loop.set_description(f"Processing batch {batch_idx + 1}")
#
#     # Your processing code here
#     print(f"Batch {batch_idx + 1}")
#     print(f"Data: {data}")
#     print(f"Labels: {labels}")
#     print("-" * 20)
print(dataset.access_num)