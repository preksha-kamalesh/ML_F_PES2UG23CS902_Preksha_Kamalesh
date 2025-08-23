# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    # Calculate entropy of the target column
    target = tensor[:, -1]
    values, counts = torch.unique(target, return_counts=True)
    probs = counts.float() / target.size(0)
    entropy = -(probs * torch.log2(probs)).sum().item()
    return round(entropy, 4)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    # Calculate weighted average entropy for each value of the attribute
    total = tensor.size(0)
    values = torch.unique(tensor[:, attribute])
    avg_info = 0.0
    for v in values:
        subset = tensor[tensor[:, attribute] == v]
        if subset.size(0) == 0:
            continue
        entropy = get_entropy_of_dataset(subset)
        weight = subset.size(0) / total
        avg_info += weight * entropy
    return round(avg_info, 4)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    # Information Gain = Entropy(S) - Avg_Info(attribute)
    entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = entropy - avg_info
    return round(info_gain, 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    # Compute information gain for all attributes except target
    num_attributes = tensor.size(1) - 1
    info_gains = {}
    for i in range(num_attributes):
        info_gains[i] = get_information_gain(tensor, i)
    best_attr = max(info_gains, key=info_gains.get)
    return info_gains, best_attr
