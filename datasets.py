from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import glob
import torch

class UnlabeledImageDataset(Dataset):  # TODO support for putting on CPU with pinned memory.
    def __init__(self, path, input_dimensions, device, channels=4, preprocessing=None, filetype="png"):
        """Loads images as a dataset without class labels.
        
        Args:
            path (str): Folder to load images from, including recursively.
            input_dimensions (tuple of (int, int)): Height and width of image files. All should be same size.
            device (str): cpu, cuda, etc.
            channels (int, optional): 1=Grayscale, 3=RGB, 4=RGB_ALPHA. Defaults to RGB_ALPHA.
            preprocessing (callable, optional): Preprocessing to be performed on images (e.g. resizing). Defaults to None.
        """
        
        super().__init__()
        assert len(input_dimensions) == 2, "Image should have 2 dimensions."
        
        assert channels in (1, 3, 4), "Only grayscale, RGB, or RGBA supported."
        modes = [torchvision.io.ImageReadMode.GRAY,
                 torchvision.io.ImageReadMode.GRAY_ALPHA,  # This is not fully supported, (pyplot doesn't like it), so we don't actually allow this.
                 torchvision.io.ImageReadMode.RGB,
                 torchvision.io.ImageReadMode.RGB_ALPHA]
        mode = modes[channels-1]
            
        files = glob.glob(path + f"/**/*.{filetype}", recursive=True)
        self.data = torch.empty((len(files), channels, input_dimensions[0], input_dimensions[1]), dtype=torch.float, device=device)
        for i, filename in enumerate(files):  # Note: making no guarantee as to order.
            self.data[i] = torchvision.io.read_image(filename, mode) / 128 - .5
        
        if preprocessing:
            self.data = preprocessing(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    
def split_ds(ds, split, batch_size):
    """Splits dataset and makes Dataloaders. First loader will be shuffled (for training), others not.
    
    Args:
        ds (Dataset): Dataset to split.
        split (tuple of floats): Percent of dataset for each split.
        batch_size (int): Dataloader batch size.
    Returns:
        [dataloader 1, dataloader 2...)]
    """
    datasets = random_split(ds, split, torch.Generator().manual_seed(23))
    loaders = []
    loaders.append(DataLoader(datasets[0], batch_size=batch_size, shuffle=True))
    for dataset in datasets[1:]:
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False))
    
    return loaders