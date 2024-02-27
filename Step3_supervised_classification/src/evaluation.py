import torch
import numpy as np
from utils import smoothing


def eval(dataloader, device, model, loss_function, class_nb):
    """
    Evaluation function.

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader
        device (torch.device): device
        model (torch.nn.Module): model
        loss_function (torch.nn.Module): loss function
        class_nb (int): number of classes

    Returns:
        eval_loss, acc, IoU, P, R, F1.
    """

    model.eval()
    eval_loss = 0.0

    # Create a dictionnary to save results per classes
    d_class_eval = dict()
    for c in range(1, class_nb):
        d_class_eval[c] = dict()

    with torch.no_grad():
        correct, total = 0, 0
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if loss_function is not None:
                loss = loss_function(outputs, targets)
                eval_loss += loss.item()  # collect loss

            predicted = torch.argmax(outputs, dim=1).tolist()[0]
            predicted = torch.tensor(smoothing(predicted)).unsqueeze(0).to(device)  # smooth predictions

            # Set total and correct (for global accuracy)
            total += targets.size(1)
            correct += (predicted == targets).sum().item()

            # Iou,P, R and F1 per classes for each video
            for c in range(1, class_nb):
                idx_target = torch.where(targets == c)[1].tolist()

                if len(idx_target) != 0:  # only consider videos where c is in ground truth
                    idx_pred = torch.where(predicted == c)[1].tolist()

                    TP = len(set(idx_target) & set(idx_pred))
                    FP = len(set(idx_pred) - set(idx_target))
                    FN = len(set(idx_target) - set(idx_pred))

                    if TP == 0:
                        IoU = 0
                        P = 0
                        R = 0
                        F1 = 0

                    else:
                        IoU = TP / len(set(idx_target + idx_pred))
                        P = TP / (TP + FP)
                        R = TP / (TP + FN)
                        F1 = 2 * (P * R) / (P + R)

                    # Save results for each video
                    try:
                        d_class_eval[c]["IoU"].append(IoU)
                        d_class_eval[c]["P"].append(P)
                        d_class_eval[c]["R"].append(R)
                        d_class_eval[c]["F1"].append(F1)
                    except:
                        d_class_eval[c]["IoU"] = [IoU]
                        d_class_eval[c]["P"] = [P]
                        d_class_eval[c]["R"] = [R]
                        d_class_eval[c]["F1"] = [F1]

    acc = correct / total  # compute accuracy

    # 1. average scores over all videos
    # 2. average scores over all classes
    IoU, P, R, F1 = 0, 0, 0, 0
    for c, dico in d_class_eval.items():
        try:
            IoU += np.mean(dico["IoU"])
            P += np.mean(dico["P"])
            R += np.mean(dico["R"])
            F1 += np.mean(dico["F1"])
        except:
            pass

    IoU = IoU / (class_nb - 1)
    P = P / (class_nb - 1)
    R = R / (class_nb - 1)
    F1 = F1 / (class_nb - 1)

    return eval_loss, acc, IoU, P, R, F1
