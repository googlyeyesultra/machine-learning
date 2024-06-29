from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import glob
import torch

class UnlabeledImageDataset(Dataset):
    def __init__(self, path, input_dimensions, device, channels=4, preprocessing=None):
        super().__init__()
        assert len(input_dimensions) == 2, "Image should have 2 dimensions."
        
        assert channels in (1, 3, 4), "Only grayscale, RGB, or RGBA supported."
        modes = [torchvision.io.ImageReadMode.GRAY,
                 torchvision.io.ImageReadMode.GRAY_ALPHA,  # This is not fully supported, (pyplot doesn't like it), so we don't actually allow this.
                 torchvision.io.ImageReadMode.RGB,
                 torchvision.io.ImageReadMode.RGB_ALPHA]
        mode = modes[channels-1]
            
        files = glob.glob(path + "/**/*.png", recursive=True)
        self.data = torch.empty((len(files), channels, input_dimensions[0], input_dimensions[1]), dtype=torch.float, device=device)
        for i, filename in enumerate(files):  # Note: making no guarantee as to order.
            self.data[i] = torchvision.io.read_image(filename, mode) / 128 - .5
        
        if preprocessing:
            self.data = preprocessing(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    
def split_ds(ds, split, batch_size):  # Assumes first split is for training, so should be shuffled. Others not.
    # Returns [dataset 1, dataset 2...], [dataloader 1, dataloader 2...)]
    datasets = random_split(ds, split, torch.Generator().manual_seed(23))
    loaders = []
    loaders.append(DataLoader(datasets[0], batch_size=batch_size, shuffle=True))
    for dataset in datasets:
        loaders.append(DataLoader(datasets[0], batch_size=batch_size, shuffle=False))
    
    return datasets, loaders