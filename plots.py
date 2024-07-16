import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def show_imgs(imgs, title=None, labels=None, grayscale=False):
    img_scale = 2
    plt.figure(figsize=(len(imgs)*img_scale, 2.2))
    
    if grayscale:
        plt.gray()

    if title:
        plt.suptitle(title)
        
    plt.gcf().set_facecolor("pink")  # Helps see alpha vs white.
    
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        if labels is not None:
            plt.title(labels[i])
        plt.axis("off")
        img = img.detach().cpu()
        img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        if not grayscale:  # Grayscale is assuming no channels dimension.
            img = img.permute(1, 2, 0)  # Move channels last.
        plt.imshow(img)
        
    plt.show()

    
def plot_stats(stats, labels, title, xlabel="Epochs", percent=False):  # Tuple of lists, tuple of strings, string.
    epochs = len(stats[0])
    for stat, label in zip(stats, labels):
        plt.plot(range(1, epochs+1), stat, label=label)
        
    plt.xlabel(xlabel)
    plt.legend()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if percent:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.gca().set_ylim([0, 1])
        
    plt.title(title)
    plt.show()