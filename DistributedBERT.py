import os
import time

import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.optim as optim
# import torch.nn.functional as F
# from torchsummary import summary
from tqdm import tqdm
import fsspec
from alluxiofs import AlluxioFileSystem, AlluxioClient
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import bisect


class BertDataset(Dataset):
    def __init__(self, alluxio_fs, tokenizer, max_length, directory_path, preprocessed_file_info, total_length,
                 read_chunk_size, output_filename):
        super(BertDataset, self).__init__()
        self.alluxio_fs = alluxio_fs
        self.directory_path = directory_path

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.preprocessed_file_info = preprocessed_file_info
        self.start_line_num_list = sorted(list(self.preprocessed_file_info.keys()))
        self.total_length = total_length
        self.read_chunk_size = read_chunk_size

        self.output_filename = output_filename  # for output recording purpose
        self.total_access = 0  # for output recording purpose

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        '''
        Map index into target line number in a specific file.
        To avoid a single big file overloading the memory, a file is read by chunk when accessing the target line.
        :param index: global index of the data point that the model trainer want to access
        :return: BERT specific tensors for the target data point
        '''

        # find the target file and the target line where the index located
        target_file_index = bisect.bisect_right(self.start_line_num_list, index) - 1
        target_file_start_line_num = self.start_line_num_list[target_file_index]
        target_file_name = self.preprocessed_file_info[target_file_start_line_num]
        target_line_index = index - target_file_start_line_num

        # load target file in memory by chunk to avoid memory overloading and then read the target line
        chunk_number = target_line_index // self.read_chunk_size
        line_within_chunk = target_line_index % self.read_chunk_size
        chunk_iterator = pd.read_csv(self.alluxio_fs.open(target_file_name, mode='r'),
                                     chunksize=self.read_chunk_size)

        for i, chunk in enumerate(chunk_iterator):
            # only the chunk will be loaded into memory each time
            # memory occupied by chunk will be freed once outside the loop
            if i == chunk_number:
                target_line = chunk.iloc[line_within_chunk]

        # process target line text for BERT use
        inputs = self.tokenizer.encode_plus(
            target_line.iloc[0],
            None,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        # record behavior to a output file
        self.total_access += 1
        with open(self.output_filename, 'a') as file:
            file.write(
                f'access to global index {index}, which is line {target_line_index} in file {target_file_name}: {target_line.iloc[0]}\n')
            file.write(f'__getitem__ total access: {self.total_access}\n')

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(target_line.iloc[1], dtype=torch.long)
        }


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        out = self.out(o2)

        return out


def finetune(epochs, dataloader, model, loss_fn, optimizer, rank, cpu_id):

    # initialize an output file to log behavior for a specific cpu
    output_filename = f"output_rank_{cpu_id}.txt"
    with open(output_filename, 'w') as file:
        file.write(f"Output content for rank {cpu_id}\n")

    # model.train()

    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
        for batch, dl in loop:

            print(f"rank {cpu_id} epoch {epoch} batch {batch}")
            append_to_output(output_filename, f"rank {cpu_id} epoch {epoch} batch {batch}")

            # ids = dl['ids'].to(rank)
            # token_type_ids = dl['token_type_ids'].to(rank)
            # mask = dl['mask'].to(rank)
            # label = dl['target'].to(rank)
            # label = label.unsqueeze(1)

            # optimizer.zero_grad()

            # output = model(
            #     ids=ids,
            #     mask=mask,
            #     token_type_ids=token_type_ids)
            # label = label.type_as(output)

            # loss = loss_fn(output, label)
            # loss.backward()
            #
            # optimizer.step()
            #
            # pred = np.where(output.cpu().detach().numpy() >= 0, 1, 0)
            # label = label.cpu().detach().numpy()
            #
            # num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
            # num_samples = pred.shape[0]
            # accuracy = num_correct / num_samples
            #
            # print(
            #     f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
            #
            # loop.set_description(f'Epoch={epoch + 1}/{epochs}')
            # loop.set_postfix(loss=loss.item(), acc=accuracy)

    return model


def preprocess(directory_path, chunk_size):
    '''
    Preprocess each file in the directory for Dataset class.
    :param directory_path:
    :param chunk_size:
    :return: processed_file_info is a dictionary that contain start_line number for each file;
            total length of all files in the directory
    '''
    fsspec.register_implementation("alluxiofs", AlluxioFileSystem, clobber=True)
    alluxio_fs = fsspec.filesystem("alluxiofs", etcd_hosts="localhost", etcd_port=2379, target_protocol="s3")

    processed_file_info = {}  # a dictionary of {start_line_number: file_name}
    next_start_line_num = 0
    total_length = 0

    all_files_info = alluxio_fs.ls(directory_path)

    # iterate each file in alluxio cache to get start line number for each file
    for file_info in all_files_info:
        file_name = file_info['name']
        processed_file_info[next_start_line_num] = file_name

        # calculate length of each file, read the csv by chunk in case the file is too big to fit into the memory
        file_length = 0
        chunk_iterator = pd.read_csv(alluxio_fs.open(file_name, mode='r'),
                                     chunksize=chunk_size)
        for chunk in chunk_iterator:
            file_length += len(chunk)

        next_start_line_num += file_length
        total_length += file_length

    return processed_file_info, total_length


def append_to_output(filename, content):
    with open(filename, 'a') as file:
        file.write(content + '\n')


def main(rank, world_size, total_epochs, batch_size, directory_path):

    # set up alluxio filesystem and load files in directory into Alluxio cache
    fsspec.register_implementation("alluxiofs", AlluxioFileSystem, clobber=True)
    alluxio_fs = fsspec.filesystem("alluxiofs", etcd_hosts="localhost", etcd_port=2379, target_protocol="s3")

    # distributed training settings
    print(f"Initializing process group for rank {rank}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    # preprocess file in directory
    preprocessed_file_info, total_length = preprocess(directory_path, 1000)

    # train BERT
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    print(f"Loading dataset on rank {rank}")

    dataset = BertDataset(alluxio_fs, tokenizer, max_length=100, directory_path=directory_path,
                          preprocessed_file_info=preprocessed_file_info, total_length=total_length,
                          read_chunk_size=1000, output_filename=f"output_rank_{rank}.txt")

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

    model = BERT().to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for param in model.module.bert_model.parameters():
        param.requires_grad = False

    model = finetune(total_epochs, dataloader, model, loss_fn, optimizer, device, rank)
    dist.destroy_process_group()
    print(f"Process group destroyed for rank {rank}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('directory_path', type=str, help='Path to the input data file')
    args = parser.parse_args()

    # initialize AlluxioClient to pull all file from S3 to alluxio
    alluxio_client = AlluxioClient(etcd_hosts="localhost")
    load_success = alluxio_client.submit_load(args.directory_path)
    print('Alluxio Load job submitted successful:', load_success)

    load_progress = "Loading datasets into Alluxio"

    while load_progress != "SUCCEEDED":
        time.sleep(5)
        progress = alluxio_client.load_progress('s3://sibyltest/BERT_test/')
        load_progress = progress[1]['jobState']
        print('Load progress:', load_progress)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
    mp.spawn(main, args=(world_size, args.total_epochs, args.batch_size, args.directory_path), nprocs=world_size,
             join=True)
