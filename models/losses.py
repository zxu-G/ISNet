import torch
import numpy as np




def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()


def log_and_append(test_log, log_prefix, amae, armse, amape):
    for i in range(12):
        log = f'{log_prefix} test data for horizon {{:d}}, Test MAE: {{:.4f}}, Test RMSE: {{:.4f}}, Test MAPE: {{:.4f}}%'
        print(log.format(i + 1, amae[i], armse[i], amape[i] * 100))
        test_log.append([i + 1, amae[i], armse[i], amape[i] * 100])
    log = f'({log_prefix} On average over 12 horizons) Test MAE: {{:.3f}} | Test RMSE: {{:.3f}} | Test MAPE: {{:.2f}}% |'
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape) * 100))
    test_log.append([13, np.mean(amae), np.mean(armse), np.mean(amape) * 100])





def AdaptiveLoss(loss, labels):

    thresholds = [0.1, 0.12, 0.2]
    weights = [0.75, 20, 1500, 3500]
    adapt_rate = [1.5, 1.75, 2.0]

    idx1 = (labels > 0) & (labels <= 0.1)
    idx2 = (labels > 0.1) & (labels <= 0.3)
    mloss = ( torch.mean(loss[idx1]) + torch.mean(loss[idx2]) ) / 2
    weights[0] = weights[0] - adapt_rate[0] * weights[0] * (thresholds[0] - mloss) if mloss < thresholds[0] else weights[0] + weights[0] * (mloss - thresholds[0])
    weights[1] = weights[1] - adapt_rate[1] * weights[1] * (thresholds[0] - mloss) if mloss < thresholds[0] else weights[1] + weights[1] * (mloss - thresholds[0])

    loss[idx1] = loss[idx1] * weights[0]
    loss[idx2] = loss[idx2] * weights[1]

    idx1 = (labels > 0.3) & (labels <= 0.6)
    mloss = torch.mean(loss[idx1])
    weights[2] = weights[2] - adapt_rate[2] * weights[2] * (thresholds[1] - mloss) if mloss < thresholds[1] else weights[2] + adapt_rate[2] * weights[2] * (mloss - thresholds[1])
    loss[idx1] = loss[idx1] * weights[2]

    idx1 = (labels > 0.6) & (labels <= 2.0)
    mloss = torch.mean(loss[idx1])
    weights[3] = weights[3] - adapt_rate[2] * weights[3] * (thresholds[2] - mloss) if mloss < thresholds[2] else weights[3] + adapt_rate[2] * weights[3] * (mloss - thresholds[2])
    loss[idx1] = loss[idx1] * weights[3]

    weights = [max(0.1, weight) for weight in weights]

    return loss


def mean_weight_Loss(loss, labels):

    mw_Loss = 0

    for i in range(15):
        idx = (labels > i*0.1) & ( labels <= (i + 1)*0.1 )

        if i<=1:
            loss_idx = loss[idx] * 0.5
        elif i>1 & i<=3:
            loss_idx = loss[idx] * 1
        elif i>3 & i<=5:
            loss_idx = loss[idx] * 4
        elif i>5 & i<= 7:
            loss_idx = loss[idx] * 8
        elif i>7:
            loss_idx = loss[idx] * 16

        if loss_idx.numel() > 0:
            mw_Loss += loss_idx.mean()

    return mw_Loss




def masked_mae(preds, labels, null_val=np.nan, weight=False):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)

    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)


    if weight:
        loss = AdaptiveLoss(loss, labels)

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return loss



def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss



def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss




def metric(pred, real, real_lat, real_lon, uni_coor=False):
    mae = masked_mae(pred,real,0.0)
    mape = masked_mape(pred,real,0.0)
    rmse = masked_rmse(pred,real,0.0)


# All
    mae1 = torch.mean(mae).item()
    mape1 = torch.mean(mape).item()
    rmse1 = torch.mean(rmse).item()

# Weak
    idx = (real > 0.0) & (real <= 0.3)
    mae2 = torch.mean(mae[idx]).item()
    mape2 = torch.mean(mape[idx]).item()
    rmse2 = torch.mean(rmse[idx]).item()

# Moderate
    idx = (real > 0.3) & (real <= 0.6)
    mae3 = torch.mean(mae[idx]).item()
    mape3 = torch.mean(mape[idx]).item()
    rmse3 = torch.mean(rmse[idx]).item()

# Strong
    idx = (real > 0.6) & (real <= 2)
    mae4 = torch.mean(mae[idx]).item()
    mape4 = torch.mean(mape[idx]).item()
    rmse4 = torch.mean(rmse[idx]).item()

    return mae1, mape1, rmse1, mae2, mape2, rmse2, mae3, mape3, rmse3, mae4, mape4, rmse4


def metric_segment(pred, real, real_lat, real_lon, ranges, segment=True, uni_coor=False):
    mae = masked_mae(pred,real,0.0)
    mape = masked_mape(pred,real,0.0)
    rmse = masked_rmse(pred,real,0.0)


    idx = (real > ranges[0]) & (real <= ranges[1])
    mae = torch.mean(mae[idx]).item()
    mape = torch.mean(mape[idx]).item()
    rmse = torch.mean(rmse[idx]).item()

    return mae,mape,rmse





def masked_huber(preds, labels, null_val=np.nan):
    crit = torch.nn.SmoothL1Loss()

    return crit(preds, labels)
