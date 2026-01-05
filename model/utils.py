import torch
import numpy as np
import os
import csv


def get_target_label_idx(labels, targets):
    """
        Get the indices of labels that are included in targets.
        :param labels: array of labels
        :param targets: list/tuple of target labels
        :return: list with indices of target labels
        """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def init_center_c(train_loader, net, device, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.hidden_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data = data.to(device)
            outputs, _ = net(data)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def write_csv_file(xp_path, file_name, head, data):
    path = os.path.join(xp_path, file_name)
    try:
        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel')

            if head is not None:
                writer.writerow(head)

            for row in data:
                writer.writerow(row)

            print("Write a CSV file to path %s Successful." % path)
    except Exception as e:
        print("Write an CSV file to path: %s, Case: %s" % (path, e))