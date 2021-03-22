from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard

def get_metrics(model, dataloader, mse_loss, max_samples=-1, device='cpu'):
    tr_batch_size = dataloader.batch_size
    samples_total = len(dataloader)
    cls_true = np.array([])
    cls_pred = np.array([])
    loss = 0
    loss_elems = 0
    seen_samples = 0
    for i, (users, movies, ratings) in enumerate(dataloader):
        users.to(device), movies.to(device), ratings.to(device)
        with torch.no_grad():
            net_out = model(users, movies)
        loss += mse_loss(net_out, ratings).cpu().item()
        loss_elems += 1
        seen_samples += len(ratings)

        #early stop (for large datasets)
        if max_samples > 0: # -1 means do not early stop
            if 0 < max_samples and max_samples < 1: # max_samples is a ratio
                if seen_samples/samples_total > max_samples:
                    break
            else: # max_samples is a number of samples
                if seen_samples > max_samples:
                    break
    if loss_elems > 0:
        loss /= loss_elems
    return {'RMSE': loss**.5, 'Loss': loss}

def train_mf(model, train_loader, test_loader,
             epochs=5, lr=1e-3,
             regularizing_params=[1.0, 1.0],
             comment='mf_train', device='cpu'):
    criterion = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)#
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pbar1 = tqdm(range(epochs+1), desc='epochs')
    pbar2 = tqdm(total=len(train_loader), desc='batches')
    tensorboard_step = 0

    with torch.utils.tensorboard.SummaryWriter(comment=comment) as writer:
        for epoch in pbar1:  # loop over the dataset multiple times
            pbar1.set_description('training')
            pbar2.reset()
            for i, (users, movies, ratings) in enumerate(train_loader):
                if epoch == 0:
                    break
                users.to(device), movies.to(device), ratings.to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(users, movies)
                loss = criterion(outputs, ratings)
                l2_reg = torch.tensor(0.)
                for reg_param, param in zip(regularizing_params,
                                            model.parameters()):
                    l2_reg += reg_param**2 * param.pow(2).mean()
                    #l1_reg += reg_param**2 * param.abs().mean()
                (loss + l2_reg).backward()
                optimizer.step()

                #metrics:
                tensorboard_step += len(ratings)
                writer.add_scalar('batch/MSELoss', 
                                  loss.item(), tensorboard_step)
                writer.add_scalar('batch/l2_reg', 
                                  l2_reg.item(), tensorboard_step)
                pbar2.set_postfix({'batch loss': loss.item()})
                pbar2.update(1)

            pbar1.set_description('metrics on train_df...')
            metrics = get_metrics(model, train_loader,
                                  criterion, max_samples=500, 
                                  device=device)
            pbar1.set_postfix(metrics)
            for metric_name, metric_value in metrics.items():
                 writer.add_scalar(f'{metric_name}/train',
                                   metric_value, epoch)

            if test_loader is not None:
                pbar1.set_description('metrics on val_df...')
                metrics = get_metrics(model, test_loader,
                                      criterion, 
                                      device=device)
                pbar1.set_postfix(metrics)
                for metric_name, metric_value in metrics.items():
                    writer.add_scalar(f'{metric_name}/validation',
                                      metric_value, epoch)
    return metrics