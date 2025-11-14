import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
from scipy import ndimage
import pdb
import time
from os.path import isfile, join
import statistics
from PIL import Image
from medpy.metric.binary import dc, hd, asd, assd
import scipy.spatial
import matplotlib.pyplot as plt
from random import random, randint, choice

labels = {0: 'Background', 1: 'Foreground'}


########## LOSS FUNCTIONS ##########

class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.0, reduction='mean'):
        """
        Tversky Focal Loss.

        Args:
            alpha: Poids pour les faux positifs.
            beta: Poids pour les faux négatifs.
            gamma: Paramètre de focalisation.
            reduction: 'mean', 'sum', ou 'none' pour contrôler la réduction de la perte.
        """
        super(TverskyFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Inputs:
            inputs: Tensor de prédictions (logits non normalisés), de forme (N, C, H, W).
            targets: Tensor d'étiquettes de vérité terrain, de forme (N, H, W) avec des valeurs entières [0, C-1].
        """
        # Convertir les cibles en one-hot
        num_classes = inputs.size(1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Appliquer le softmax pour obtenir des probabilités
        inputs_soft = F.softmax(inputs, dim=1)

        # Calcul des termes pour la Tversky index
        dims = (0, 2, 3)  # Somme sur batch et dimensions spatiales
        true_pos = torch.sum(inputs_soft * targets_onehot, dim=dims)
        false_neg = torch.sum(targets_onehot * (1 - inputs_soft), dim=dims)
        false_pos = torch.sum((1 - targets_onehot) * inputs_soft, dim=dims)

        # Tversky index
        tversky_index = true_pos / (true_pos + self.alpha * false_pos + self.beta * false_neg + 1e-7)

        # Focal Tversky Loss
        tversky_loss = (1 - tversky_index) ** self.gamma

        # Réduction
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes=4):
        """
        Dice Loss.

        Args:
            num_classes: Nombre de classes (y compris l'arrière-plan).
            epsilon: Valeur pour éviter la division par zéro.
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        Inputs:
            inputs: Tensor de prédictions (logits non normalisés), de forme (N, C, H, W).
            targets: Tensor d'étiquettes de vérité terrain, de forme (N, H, W) avec des valeurs entières [0, C-1].
        """
        # Convertir les cibles en one-hot
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Appliquer le softmax
        inputs_soft = F.softmax(inputs, dim=1)

        # Calcul des termes pour le Dice coefficient
        dims = (0, 2, 3)  # Somme sur batch et dimensions spatiales
        intersection = torch.sum(inputs_soft * targets_onehot, dim=dims)
        cardinality = torch.sum(inputs_soft + targets_onehot, dim=dims)

        # Dice coefficient
        dice = 2.0 * intersection / (cardinality + 1e-7)

        # Dice Loss
        dice_loss = 1 - dice

        return dice_loss.mean()
    

######### METRICS, STATISTICS & PLOTS #########

def compute_metrics(predictions, ground_truth, num_classes):
    """
    Compute evaluation metrics for a batch of predictions and ground truths.

    Parameters:
    - predictions: Tensor of predicted segmentation masks (batch_size, height, width)
    - ground_truth: Tensor of ground truth segmentation masks (batch_size, height, width)
    - num_classes: The total number of classes

    Returns:
    - metrics: Dictionary containing IoU, Precision, Recall, F1, and DSC for each class
    """
    metrics = {
        'IoU': {cls: [] for cls in range(1, num_classes)},
        'Precision': {cls: [] for cls in range(1, num_classes)},
        'Recall': {cls: [] for cls in range(1, num_classes)},
        'DSC': {cls: [] for cls in range(1, num_classes)},
    }

    for cls in range(1, num_classes):  # Exclude the background class (0)
        # Create binary masks for the current class
        pred_binary = (predictions == cls).float()
        gt_binary = (ground_truth == cls).float()

        # Flatten tensors to 1D
        pred_flat = pred_binary.view(-1)
        gt_flat = gt_binary.view(-1)

        # Compute True Positives (TP), False Positives (FP), False Negatives (FN)
        TP = (pred_flat * gt_flat).sum()
        FP = (pred_flat * (1 - gt_flat)).sum()
        FN = ((1 - pred_flat) * gt_flat).sum()

        # Compute Metrics
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)
        iou = TP / (TP + FP + FN + 1e-7)
        dsc = 2 * TP / (2 * TP + FP + FN + 1e-7)

        # Store metrics
        metrics['Precision'][cls].append(precision.item())
        metrics['Recall'][cls].append(recall.item())
        metrics['IoU'][cls].append(iou.item())
        metrics['DSC'][cls].append(dsc.item())

    return metrics


def plot_metrics(modelName, num_epochs, num_classes):
    """
    Plots the IoU, Precision, Recall, and F1 metrics over epochs for each class.

    Parameters:
    - modelName: The name of the model (e.g., 'UNet')
    - num_epochs: The total number of epochs
    - num_classes: The total number of classes
    """

    directory = f'Results/Statistics/{modelName}'
    metric_directory = os.path.join(directory, 'metrics')
    metrics_names = ['IoU', 'Precision', 'Recall', 'DSC']
    epoch_range = range(num_epochs)

    for metric_name in metrics_names:
        plt.figure(figsize=(10, 6))
        for cls in list(range(1, num_classes)): # Exclude the background class (0)
            metric_values = []
            for epoch in epoch_range:
                metrics_file = os.path.join(metric_directory, f'metrics_epoch_{epoch}.npy')
                if os.path.exists(metrics_file):
                    # Load the metrics for this epoch
                    metrics = np.load(metrics_file, allow_pickle=True).item()
                    class_metrics = metrics[metric_name][cls]
                    # Compute the mean metric value for the class at this epoch
                    mean_metric = np.mean(class_metrics)
                    metric_values.append(mean_metric)
                else:
                    print(f"Metrics file for epoch {epoch} not found.")
                    break
            if metric_values:
                # Plot the metric values over epochs for this class
                plt.plot(epoch_range[:len(metric_values)], metric_values, label=f'Class {cls}')
        plt.title(f'{metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        plot_save_path = os.path.join('Results/Plots', modelName)
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        plt.savefig(os.path.join(plot_save_path, f'{metric_name}.png'))
        plt.close()

    # Delete metric values (all files in the metrics directory)
    for file in os.listdir(metric_directory):
        os.remove(os.path.join(metric_directory, file))


def plot_losses(modelName):
    """
    Plots the training and validation losses over epochs.
    
    Parameters:
    - modelName: The name of the model (e.g., 'UNet')
    """

    directory = f'Results/Statistics/{modelName}'

    # Load training and validation losses
    train_losses = np.load(os.path.join(directory, 'Train_Losses.npy'))
    val_losses = np.load(os.path.join(directory, 'Val_Losses.npy'))

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a file
    plot_save_path = os.path.join('Results/Plots', modelName)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    plt.savefig(os.path.join(plot_save_path, 'Losses.png'))
    plt.close()

    # Delete loss values (all files in the losses directory)
    os.remove(os.path.join(directory, 'Train_Losses.npy'))
    os.remove(os.path.join(directory, 'Val_Losses.npy'))


def compute_class_pixel_counts(dataset, num_classes):
    """
    Computes the number of pixels for each class in the given dataset.

    Parameters:
    - dataset: The dataset object (e.g., train_set_full)
    - num_classes: The total number of classes

    Returns:
    - class_pixel_counts: A list containing the pixel count for each class
    """
    class_pixel_counts = [0] * num_classes  # Initialize counts for each class

    for idx in range(len(dataset)):
        # Get the label (mask) for the current sample
        _, label, _ = dataset[idx]  # Assuming dataset[idx] returns (image, label, img_name)

        # Convert label to segmentation classes
        label_np = np.array(label)
        # Convert to class indices (0 to num_classes - 1)
        label_classes = getTargetSegmentation(torch.tensor(label_np)).numpy()

        # Count pixels for each class
        for cls in range(num_classes):
            class_pixel_counts[cls] += np.sum(label_classes == cls)

    return class_pixel_counts


def compute_class_weights(pixel_counts):
    """
    Computes class weights inversely proportional to class pixel counts.

    Parameters:
    - pixel_counts: A list or array containing pixel counts for each class

    Returns:
    - class_weights: A torch tensor containing weights for each class
    """
    # Convert pixel counts to numpy array
    pixel_counts = np.array(pixel_counts, dtype=np.float32)

    # Avoid division by zero
    epsilon = 1e-6
    pixel_counts = pixel_counts + epsilon

    # Compute inverse frequencies
    inv_freq = 1.0 / pixel_counts

    # Normalize the weights
    class_weights = inv_freq / np.sum(inv_freq) * len(pixel_counts)

    # Convert to torch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if torch.cuda.is_available():
        class_weights = class_weights.cuda()

    return class_weights


def computeDSC(pred, gt):
    """
    Compute the Dice Similarity Coefficient (DSC) for a batch of predictions and ground truth masks.
    
    Parameters:
    - pred: The model's predicted masks, of shape (batch_size, 1, H, W)
    - gt: The ground truth masks, of shape (batch_size, 1, H, W)

    Returns:
    - DSC: The average Dice Similarity Coefficient for the batch
    """

    dscAll = []
    for i in range(pred.shape[0]):
        dscAll.append(dc(pred[i].cpu().numpy(), gt[i].cpu().numpy()))
    return torch.tensor(statistics.mean(dscAll))


def display_segmented_images(images, labels: None, pred_masks, num_images=5):
    """
    Displays the original images, ground truth labels, and predicted masks side by side.

    Parameters:
    - images: The original images tensor, of shape (N, C, H, W)
    - labels: The ground truth labels tensor, of shape (N, 1, H, W)
    - pred_masks: The predicted masks tensor, of shape (N, 1, H, W)
    - num_images: The number of images to display (default is 5)
    """

    images = images.cpu().numpy()
    if labels is not None: 
        labels = labels.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()

    nb_plots = 3 if labels is not None else 2

    for i in range(min(num_images, len(images))):
        fig, axs = plt.subplots(1, nb_plots, figsize=(nb_plots*5, 5))
        axs[0].imshow(images[i].transpose(1, 2, 0))  # Image originale
        axs[0].set_title("Image originale")
        axs[0].axis('off')

        axs[1].imshow(pred_masks[i].squeeze(), cmap='gray')  # Masque prédit
        axs[1].set_title("Masque prédit")
        axs[1].axis('off')

        # Suppression de la dimension supplémentaire
        if labels is not None:
            axs[2].imshow(labels[i].squeeze(), cmap='gray')  # Masque d'étiquette
            axs[2].set_title("Masque réel")
            axs[2].axis('off')

        plt.show()


def display_dataset_samples(dataset, num_samples=5):
    """
    Displays image and label pairs from a given dataset.

    Parameters:
    - dataset: The dataset object (e.g., new_pseudo_dataset)
    - num_samples: Number of samples to display (default is 5)
    """

    for idx in range(min(num_samples, len(dataset))):
        img, mask, img_name = dataset[idx]

        # Convert tensors to numpy arrays for visualization
        img_np = img.numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        mask_np = mask.numpy().squeeze()  # Remove channel dimension if present

        # Plot the image and mask
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_np, cmap='gray')
        axs[0].set_title(f"Image: {os.path.basename(img_name)}")
        axs[0].axis('off')

        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Pseudo-Label")
        axs[1].axis('off')

        plt.show()



######### DATA PROCESSING #########

def augment(image, mask=None, transformations_dict=None):
    """
    Applies random augmentations to the image and mask (tensors).

    Parameters:
    - image: The input image tensor (C, H, W)
    - mask: The input mask tensor (C, H, W) (optional)
    - transformations_dict: A dictionary to store the transformations applied (optional)

    Returns:
    - image: The augmented image tensor
    - mask: The augmented mask tensor
    - transformations_dict: A dictionary containing the transformations applied
    """
    if transformations_dict is None:
        transformations_dict = {}
    
    # Randomly decide whether to apply horizontal flip
    if random() > 0.5:
        image = TF.hflip(image)
        if mask is not None:
            mask = TF.hflip(mask)
        transformations_dict['hflip'] = True
    
    # Randomly decide whether to apply vertical flip
    if random() > 0.5:
        image = TF.vflip(image)
        if mask is not None:
            mask = TF.vflip(mask)
        transformations_dict['vflip'] = True
    
    # Randomly decide whether to apply rotation
    if random() > 0.5:
        angle = choice([0, 90, 180, 270])
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        transformations_dict['rotate'] = angle
    
    return image, mask, transformations_dict


# Invert the transformations applied to the mask using the transformations dictionary
def de_transform_mask(mask, transformations_dict):
    """
    Inverts the transformations applied to the mask (tensor).

    Parameters:
    - mask: The input mask tensor (H, W)
    - transformations_dict: A dictionary containing the transformations applied

    Returns:
    - mask: The de-transformed mask tensor
    """
    # Ensure mask is float32 tensor
    mask = mask.float()
    # Invert transformations in reverse order
    transformations = list(transformations_dict.items())
    for transformation_name, params in reversed(transformations):
        if transformation_name == 'rotate':
            angle = -params  # Invert the rotation
            # Add channel dimension if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            mask = mask.squeeze(0)
        elif transformation_name == 'vflip':
            mask = TF.vflip(mask)
        elif transformation_name == 'hflip':
            mask = TF.hflip(mask)
    # Convert back to long (integer) tensor
    return mask.long()


# Used for the pseudo-labeling process, to check if the model is confident in its predictions
def compute_mean_iou(mask1, mask2, num_classes):
    """
    Computes the mean Intersection over Union (IoU) for a pair of masks.

    Parameters:
    - mask1: The first mask tensor, of shape (H, W)
    - mask2: The second mask tensor, of shape (H, W)
    - num_classes: The total number of classes

    Returns:
    - mean_iou: The mean IoU value for all classes
    """

    iou_per_class = []
    for cls in range(1, num_classes):  # Exclude background class 0
        mask1_cls = (mask1 == cls).float()
        mask2_cls = (mask2 == cls).float()

        # Flatten
        mask1_flat = mask1_cls.view(-1)
        mask2_flat = mask2_cls.view(-1)

        # Compute intersection and union
        intersection = (mask1_flat * mask2_flat).sum()
        union = mask1_flat.sum() + mask2_flat.sum() - intersection

        if union.item() == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union

        iou_per_class.append(iou.item())

    mean_iou = sum(iou_per_class) / len(iou_per_class)
    return mean_iou


def predToSegmentation(pred):
    """
    Converts the model's predictions to segmentation classes.

    Parameters:
    - pred: The model's predicted masks (logits), of shape (N, num_classes, H, W)

    Returns:
    - segmentation_classes: The predicted segmentation classes, of shape (N, H, W)
    """

    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def getTargetSegmentation(batch):
    """
    Converts the target masks to segmentation classes.

    Parameters:
    - batch: The target masks tensor, of shape (N, 1, H, W) with values in {0, 0.33333334, 0.6666667, 0.94117647}

    Returns:
    - segmentation_classes: The target segmentation classes, of shape (N, H, W) with values in [0, num_classes - 1]
    """

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


def classIndexToGrayscale(class_indices):
    """
    Converts class indices back to grayscale values.

    Parameters:
    - class_indices: Tensor of class indices (0, 1, 2, 3)

    Returns:
    - grayscale_values: Tensor of grayscale values (0.0, 0.33333334, 0.6666667, 0.94117647)
    """
    denom = 0.33333334  # Same as in getTargetSegmentation
    grayscale_values = class_indices.float() * denom
    # Handle the case where class index 3 should map to 0.94117647 instead of 1.0
    grayscale_values[class_indices == 3] = 0.94117647
    return grayscale_values


def to_var(x):
    """
    Converts a tensor to a PyTorch Variable.

    Parameters:
    - x: The input tensor

    Returns:
    - x: The input tensor as a PyTorch Variable
    """

    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class MaskToTensor(object):
    """
    Converts a PIL Image to a PyTorch tensor.

    Parameters:
    - img: The input image
    """

    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()



######### OTHERS #########

def DicesToDice(Dices):
    """
    Compute the Dice coefficient from a list of Dice coefficients.

    Parameters:
    - Dices: A list of Dice coefficients

    Returns:
    - Dice: The average Dice coefficient
    """

    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def inference(net, img_batch, modelName, class_weights=None):
    """
    Perform inference on a batch of images using the given model.

    Parameters:
    - net: The model to use for inference
    - img_batch: The batch of images to process
    - modelName: The name of the model

    Returns:
    - The mean loss value for the batch
    - The mean Dice score for the batch
    """
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax(dim=1).cuda()
    if class_weights is not None:
        CE_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
    else:
        CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    dice_scores = []
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)
        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())

        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        # Compute Dice score
        dice = computeDSC(masks, labels)
        dice_scores.append(dice.cpu().data.numpy())

        # Original path saving combined result
        combined_path = os.path.join('./Results/Images/', modelName)
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)

        # Save the combined image, label, and prediction batch
        torchvision.utils.save_image(
            torch.cat([images.data, labels.data, masks.unsqueeze(1).float() / 3.0]),
            os.path.join(combined_path, f'{i}.png'),
            padding=0
        )

        # Additional path for individual predictions
        pred_path = os.path.join('./Results/Predictions/', modelName)
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        # Save each predicted mask individually
        for idx in range(masks.shape[0]):
            # Extract filename from path
            base_name = os.path.basename(img_names[idx])       # e.g. "example.jpg"
            base_name_no_ext = os.path.splitext(base_name)[0]  # e.g. "example"
            mask_filename = f"{base_name_no_ext}_pred.png"

            # Save the predicted mask
            torchvision.utils.save_image(
                masks[idx].unsqueeze(0).float() / 3.0,
                os.path.join(pred_path, mask_filename),
                padding=0
            )

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    # Compute the average loss and Dice score for the batch
    losses = np.asarray(losses)
    average_loss = losses.mean()
    average_dice = np.mean(dice_scores)
    
    return average_loss, average_dice



def getImageImageList(imagesFolder):
    """
    Get the list of images in the given folder.

    Parameters:
    - imagesFolder: The folder containing the images

    Returns:
    - imageNames: The list of image names
    """
    
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames