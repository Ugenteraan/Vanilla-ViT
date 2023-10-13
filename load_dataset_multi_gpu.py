'''Module to load & transform dataset.
'''

import deeplake
import torch
from torchvision import transforms

import cred
import cfg




class LoadDeeplakeDataset:
    '''Load a dataset from the Deeplake API.
    '''

    def __init__(self, token, deeplake_ds_name, batch_size, shuffle, num_workers, world_size, rank, pin_memory=True, mode='train'):
        '''Param init.
        '''

        self.token = token
        self.deeplake_ds_name = deeplake_ds_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank


    def collate_fn(self, batch_data):
        '''Custom collate function to preprocess the batch dataset.
        '''
        return {
                'images': torch.stack([x['images'] for x in batch_data]),
                'labels': torch.stack([torch.from_numpy(x['labels']) for x in batch_data])
            }

    @staticmethod
    def training_transformation():

        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])




    @staticmethod
    def testing_transformation():
        return  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])

    def __call__(self):

        places205_dataset = deeplake.load(self.deeplake_ds_name, token=self.token)
        sampler = torch.utils.data.distributed.DistributedSampler(places205_dataset, num_replicas=self.world_size, rank=self.rank)

        if self.mode == 'train':
            dataloader = places205_dataset.pytorch(batch_size=self.batch_size, distributed=True, shuffle=self.shuffle, num_workers=self.num_workers, transform={'images':self.training_transformation(), 'labels':None}, collate_fn=self.collate_fn, decode_method={'images':'pil'}, sampler=sampler)
        else:
            dataloader = places205_dataset.pytorch(batch_size=self.batch_size, distributed=False, shuffle=self.shuffle, num_workers=self.num_workers, transform={'images':self.testing_transformation(), 'labels':None}, collate_fn=self.collate_fn, decode_method={'images':'pil'}) #no need sampler here since we want to test the model with all the test data.
        # dataloader = None
        # if self.mode == 'train':
        #     dataloader = places205_dataset.dataloader().transform({'images':self.training_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).num_workers(self.num_workers).pin_memory(self.pin_memory).sampler(train_sampler).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})
        # else:
        #     dataloader = places205_dataset.dataloader().transform({'images':self.testing_transformation(), 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).num_workers(self.num_workers).pin_memory(self.pin_memory).pytorch(collate_fn=self.collate_fn, decode_method={'images':'pil'})

        return dataloader



















