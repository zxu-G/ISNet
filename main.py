
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
import torch
torch.set_num_threads(5)
import pickle 

from torch import nn
from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import ISNet
import yaml
import setproctitle
import torch.distributed
import os




def try_all_gpus():
    device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return device


def main(**kwargs):

    set_config(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='S4', help='Dataset name.')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)   # DDP

    args = parser.parse_args()

    # DDP Initialization
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')


    config_path = "configs/" + args.dataset + ".yaml"
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        
    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['data_dir'].split("/")[-1]


    save_path       = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"             # the best model
    save_path_resume= 'output/' + config['start_up']['model_name'] + "_" + dataset_name + "_resume.pt"      # the resume model
    load_pkl        = config['start_up']['load_pkl']
    model_name      = config['start_up']['model_name']

    setproctitle.setproctitle("{0}.{1}@S4".format(model_name, dataset_name))

# ========================== load dataset ====================== #
    if load_pkl:            # default: False
        t1   = time.time()
        dataloader  = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        t2  = time.time()
        print("Load dataset: {:.2f}s...".format(t2-t1))
    else:
        t1   = time.time()
        batch_size  = config['model_args']['batch_size']

        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)

        t2  = time.time()
        print("Load dataset: {:.2f}s...".format(t2-t1))

    scaler        = dataloader['scaler']                      # re_max_min_normalization

    all_min = pickle.load(open("datasets/{0}/vmin.pkl".format(dataset_name), 'rb'))  # todo
    all_max = pickle.load(open("datasets/{0}/vmax.pkl".format(dataset_name), 'rb'))
    _max = all_max['S4']
    _min = all_min['S4']



# ================================ Hyper Parameters ================================= #
    # model parameters
    model_args  = config['model_args']
    model_args['device'] = device

    model_args['dataset']       = dataset_name
    model_args['S4_max'] = _max
    model_args['S4_min'] = _min
    model_args['feature_args'] = config['feature_args']
    model_args['feature_num'] = sum(value == True for value in model_args['feature_args'].values()) - 1



    # training strategy parametes
    optim_args                  = config['optim_args']
    optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])

# ============================= Model and Trainer ============================= #
    # log
    logger  = TrainLogger(model_name, dataset_name)
    log_path = logger.dir_path
    logger.print_optim_args(optim_args)

    # init the model
    # model = ISNet(**model_args).to(device[0])       # single

    # training init: resume model & load parameters
    mode = config['start_up']['mode']
    # mode = 'test'

    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if mode == 'test':
        model = torch.load('output/Full_model.pt', map_location='cuda:0')        # resume best
        model = model.cuda()

    else:
        if mode == 'resume':
            resume_epoch = config['start_up']['resume_epoch']
            model = load_model(model, save_path_resume)
        else:       # scratch
            model = ISNet(**model_args).cuda()                       # DDP
            resume_epoch = 0

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)

    # get a trainer
    engine  = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # begin training:
    train_time  = []
    val_time    = []

    print("Whole trainining iteration is " + str(len(dataloader['train_loader'])))

    batch_num   = resume_epoch * len(dataloader['train_loader'])     # batch number

    engine.set_resume_lr_and_cl(resume_epoch, batch_num)


# =============================================================== Training ================================================================= #
    Train_Val_log = []
    Test_log = []

    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            # train a epoch
            time_train_start    = time.time()

            current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
            train_loss = []
            train_mape = []
            train_rmse = []

            for itera, (x, y) in enumerate(dataloader['train_loader']):
                trainx = x.cuda(non_blocking=True)  # [batch_size, history_len, num_nodes, num_feats]
                trainy = y.cuda(non_blocking=True)

                mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=_max, _min=_min, lat_num=config['model_args']['lat_num'],
                                               lon_num=config['model_args']['lon_num'], loss_weight = config['model_args']['loss_weight'])
                print("{0}: {1}".format(itera, mae), end='\r')
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
            time_train_end      = time.time()
            train_time.append(time_train_end - time_train_start)

            current_learning_rate = engine.optimizer.param_groups[0]['lr']

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()
            # record history loss
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)


# =============================================================== Validation ================================================================= #
            time_val_start      = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, = engine.eval(device, dataloader['val_loader'], model_name, _max=_max, _min=_min, lat_num=config['model_args']['lat_num'],
                                                                 lon_num=config['model_args']['lon_num'])

            time_val_end        = time.time()
            val_time.append(time_val_end - time_val_start)

            curr_time   = str(time.strftime("%d-%H-%M", time.localtime()))
            log = 'Current Time: ' + curr_time + ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} | Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAPE: {:.4f} | LR: {:.6f}'

            print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mape, current_learning_rate))

            ## Early stopping
            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
# =============================================================== Test =================================================================
            test_log = engine.test(model, save_path_resume, device, dataloader, scaler, model_name, _max=_max, _min=_min, loss=engine.loss,
                                   dataset_name=dataset_name, lat_num=config['model_args']['lat_num'], lon_num=config['model_args']['lon_num'],
                                   feature_args=model_args['feature_args'])

            Train_Val_log.append([epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mape, current_learning_rate])
            Test_log.append(test_log)
            with open(log_path + '/Train_Val_log.pkl', 'wb') as file:
                pickle.dump(Train_Val_log, file)
            with open(log_path + '/Test_log.pkl', 'wb') as file:
                pickle.dump(Test_log, file)

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    else:

        test_log = engine.test(model, save_path_resume, device, dataloader, scaler, model_name, save_model_parameter=False, save_full_model=True, save_test_data=True, save_feature_emb=True,
                               _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name, all_max=all_max, all_min=all_min,
                               lat_num=config['model_args']['lat_num'], lon_num=config['model_args']['lon_num'], feature_args=config['feature_args'])




if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end   = time.time()
    print("Total time spent: {0}".format(t_end - t_start))
