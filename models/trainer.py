
import numpy as np
import torch
import pickle
import torch.optim as optim
from torchinfo.torchinfo import summary
from sklearn.metrics import mean_absolute_error
from torch import nn


from utils.train import data_reshaper, save_model
from .losses import masked_mae, masked_rmse, masked_mape, metric, metric_segment, log_and_append, mean_weight_Loss



class trainer():
    def __init__(self, scaler, model, **optim_args):
        self.model  = model         # init model
        self.scaler = scaler        # data scaler
        self.output_seq_len = optim_args['output_seq_len']  # output sequence length
        self.print_model_structure = optim_args['print_model']

        # training strategy parametes
        ## adam optimizer
        self.lrate  =  optim_args['lrate']
        self.wdecay = optim_args['wdecay']
        self.eps    = optim_args['eps']
        ## learning rate scheduler
        self.if_lr_scheduler    = optim_args['lr_schedule']
        self.lr_sche_steps      = optim_args['lr_sche_steps']
        self.lr_decay_ratio     = optim_args['lr_decay_ratio']
        ## curriculum learning
        self.if_cl          = optim_args['if_cl']
        self.cl_steps       = optim_args['cl_steps']
        self.cl_len = 0 if self.if_cl else self.output_seq_len
        ## warmup
        self.warm_steps     = optim_args['warm_steps']

        # Adam optimizer
        self.optimizer      = optim.Adam(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay, eps=self.eps)
        # learning rate scheduler
        self.lr_scheduler   = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_sche_steps, gamma=self.lr_decay_ratio) if self.if_lr_scheduler else None
        
        # loss
        self.loss   = masked_mae
        self.clip   = 5             # gradient clip
    
    def set_resume_lr_and_cl(self, epoch_num, batch_num):
        if batch_num == 0:
            return
        else:
            for _ in range(batch_num):
                # curriculum learning
                if _ < self.warm_steps:   # warmupping
                    self.cl_len = self.output_seq_len
                elif _ == self.warm_steps:
                    # init curriculum learning
                    self.cl_len = 1
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lrate
                else:
                    # begin curriculum learning
                    if (_ - self.warm_steps) % self.cl_steps == 0 and self.cl_len < self.output_seq_len:
                        self.cl_len += int(self.if_cl)
            print("resume training from epoch{0}, where learn_rate={1} and curriculum learning length={2}".format(epoch_num, self.lrate, self.cl_len))

    def print_model(self, **kwargs):
        if self.print_model_structure and int(kwargs['batch_num'])==0:
            summary(self.model, input_data=input)
            parameter_num = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.shape)
                tmp = 1
                for _ in param.shape:
                    tmp = tmp*_
                parameter_num += tmp
            print("Parameter size: {0}".format(parameter_num))





    def train(self, input, real_val, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        self.print_model(**kwargs)

        predict, dif_loss_sum, bg_loss_sum  = self.model(input, real_val)
        predict  = predict.transpose(1,2)

        # curriculum learning
        if kwargs['batch_num'] < self.warm_steps:   # warmupping
            self.cl_len = self.output_seq_len
        elif kwargs['batch_num'] == self.warm_steps:
            # init curriculum learning
            self.cl_len = 1
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lrate
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.lrate))
        else:
            # begin curriculum learning
            if (kwargs['batch_num'] - self.warm_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)
        # scale data and calculate loss
        real_lat = real_val[:, :, :, kwargs["lat_num"]]
        real_lon = real_val[:, :, :, kwargs["lon_num"]]
        if kwargs['_max'] is not None:  # traffic flow
            predict     = self.scaler(predict, kwargs["_max"], kwargs["_min"])
            real_val    = self.scaler(real_val[:,:,:,0], kwargs["_max"], kwargs["_min"])

        mae_loss = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :], 0.0, weight=kwargs["loss_weight"])


        mae_loss = torch.mean(mae_loss)        # Adding reconstruction loss to the training set only

        loss = mae_loss + dif_loss_sum  + bg_loss_sum

        loss.backward()


        # gradient clip and optimization
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # metrics
        mape = masked_mape(predict,real_val,0.0)
        rmse = masked_rmse(predict,real_val,0.0)
        mape = torch.mean(mape)
        rmse = torch.mean(rmse)

        return mae_loss.item(), mape.item(), rmse.item()



    def eval(self, device, val_data, model_name, **kwargs):

        valid_loss = []
        valid_mape = []
        valid_rmse = []
        self.model.eval()
        for itera, (x, y) in enumerate(val_data):
            valx   = x.to(device)
            valy   = y.to(device)

            output, _, _  = self.model(valx, valy)
            output  = output.transpose(1,2)
            
        # scale data
            real_lat = valy[:, :, :, kwargs["lat_num"]]
            real_lon = valy[:, :, :, kwargs["lon_num"]]
            if kwargs['_max'] is not None:
                ## inverse transform for both predict and real value.
                predict = self.scaler(output, kwargs["_max"], kwargs["_min"])
                real_val= self.scaler(valy[..., 0], kwargs["_max"], kwargs["_min"])

            # metrics
            loss = self.loss(predict, real_val, 0.0)
            mape = masked_mape(predict,real_val,0.0)
            rmse = masked_rmse(predict,real_val,0.0)

            loss = torch.mean(loss).item()
            mape = torch.mean(mape).item()
            rmse = torch.mean(rmse).item()

            print("test: {0}".format(loss), end='\r')

            valid_loss.append(loss)
            valid_mape.append(mape)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss,mvalid_mape,mvalid_rmse


    @staticmethod
    def test(model, save_path_resume, device, dataloader, scaler, model_name, save_model_parameter=True, save_full_model=True, save_test_data=False, save_feature_emb=False, **kwargs):
        # test
        model.eval()
        outputs = []
        realy   = torch.Tensor(dataloader['y_test']).to(device)

        y_list  = []
        mask_list = []
        for itera, (x, y) in enumerate(dataloader['test_loader']):
            testx   = x.to(device)
            testy   = y.to(device)

            with torch.no_grad():
                preds, _, _   = model(testx, testy)

            testy = testy.transpose(1, 2)
            outputs.append(preds)
            y_list.append(testy)

        realy   = realy.transpose(1, 2)
        yhat    = torch.cat(outputs,dim=0)[:realy.size(0),...]
        y_list  = torch.cat(y_list, dim=0)[:realy.size(0),...]

        real_lat = realy[:, :, :, kwargs["lat_num"]]
        real_lon = realy[:, :, :, kwargs["lon_num"]]


        # scale data
        if save_test_data:
            lat = scaler(realy[:, :, :, kwargs["lat_num"]], kwargs["all_max"]['Lat'], kwargs["all_min"]['Lat'])
            lon = scaler(realy[:, :, :, kwargs["lon_num"]], kwargs["all_max"]['Lon'], kwargs["all_min"]['Lon'])
            realy   = scaler(realy[:, :, :, 0], kwargs["_max"], kwargs["_min"])
            yhat    = scaler(yhat, kwargs["_max"], kwargs["_min"])

            test_data = [yhat, realy, lat, lon]
            test_data = [_.cpu().detach().numpy() for _ in test_data]
            # test_data = test_data.cpu()
            with open('output/test_output.pkl', 'wb') as file:
                pickle.dump(test_data, file)
        else:
            realy   = scaler(realy[:, :, :, 0], kwargs["_max"], kwargs["_min"])
            yhat    = scaler(yhat, kwargs["_max"], kwargs["_min"])


        # summarize the results.
        amae, amape, armse, test_log  = [], [], [], []           # all
        amae1, amape1, armse1, test_log1 = [], [], [], []        # w
        amae2, amape2, armse2, test_log2 = [], [], [], []        # m
        amae3, amape3, armse3, test_log3 = [], [], [], []        # s

        # All
        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred    = yhat[:,:,i:(i+1)]
            real    = realy[:,:,i:(i+1)]
            rlat    = real_lat[:,:,i:(i+1)]
            rlon    = real_lon[:, :, i:(i+1)]
            metrics = metric(pred, real, rlat, rlon)

            log = 'ALL-[0.0, 1.3] Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}%'
            print(log.format(i + 1, metrics[0], metrics[2], metrics[1] * 100))
            amae.append(metrics[0])  # mae
            amape.append(metrics[1])  # mape
            armse.append(metrics[2])  # rmse
            test_log.append([i + 1, metrics[0], metrics[2], metrics[1] * 100])

            amae1.append(metrics[3])  # mae
            amape1.append(metrics[4])  # mape
            armse1.append(metrics[5])  # rmse
            test_log1.append([i + 1, metrics[3], metrics[5], metrics[4] * 100])

            amae2.append(metrics[6])  # mae
            amape2.append(metrics[7])  # mape
            armse2.append(metrics[8])  # rmse
            test_log2.append([i + 1, metrics[6], metrics[8], metrics[7] * 100])

            amae3.append(metrics[9])  # mae
            amape3.append(metrics[10])  # mape
            armse3.append(metrics[11])  # rmse
            test_log3.append([i + 1, metrics[9], metrics[11], metrics[10] * 100])


        log = '([ALL] On average over 12 horizons) Test MAE: {:.3f} | Test RMSE: {:.3f} | Test MAPE: {:.2f}% |'
        print(log.format(np.mean(amae),np.mean(armse),np.mean(amape) * 100))
        test_log.append([13, np.mean(amae), np.mean(armse), np.mean(amape) * 100])

        # Weak
        log_and_append(test_log1, 'Weak-[0.0, 0.3]', amae1, armse1, amape1)

        # Moderate
        log_and_append(test_log2, 'Moderate-[0.3, 0.6]', amae2, armse2, amape2)

        # Strong
        log_and_append(test_log3, 'Strong-[0.6, 1.3]', amae3, armse3, amape3)


        test_log = {'all': test_log, 'weak': test_log1, 'moderate': test_log2, 'strong': test_log3}


        if save_model_parameter:
            save_model(model, save_path_resume)

        if save_full_model:
            torch.save(model.module, 'output/Full_model.pt')     # DDP

        if save_feature_emb:

            emb_names = [key for key, value in kwargs["feature_args"].items() if value is True]
            emb_names.remove('S4')
            feature_emb = [getattr(model.module, f'{name}_emb') for name in emb_names]
            feature_emb = [_.cpu().detach().numpy() for _ in feature_emb]

            data = {'feature_emb': feature_emb, 'emb_names': emb_names}
            with open('output/feature_emb.pkl', 'wb') as file:
                pickle.dump(data, file)



        return test_log