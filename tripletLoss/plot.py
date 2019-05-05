import torch
import numpy as np


def extract_embeddings(dataloader, model, embedding_len):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), embedding_len)) # same as faceNet
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def plot_embeddings(embeddings, target, classes, writer, tag):
    """plot embeddings in tensorboard, observe using pca or tSNE
    """
    meta = []

    for i in target:
        meta.append(str(i)+ '-'+ classes[int(i)])

    writer.add_embedding(embeddings, metadata=meta, tag=tag)


#def plot_embeddings(embeddings, targets, classes, colors, xlim=None, ylim=None):
#    plt.figure(figsize=(10,10))
#    for i in range(10):
#        inds = np.where(targets==i)[0]
#        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
#    if xlim:
#        plt.xlim(xlim[0], xlim[1])
#    if ylim:
#        plt.ylim(ylim[0], ylim[1])
#    plt.legend(classes)
#    plt.show()
# visulization

#mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#          '#bcbd22', '#17becf']
#
#train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
#    plot_embeddings(train_embeddings_tl, train_labels_tl, mnist_classes, colors)
#    val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
#    plot_embeddings(val_embeddings_tl, val_labels_tl, mnist_classes, colors)
