import random
import sys

sys.path.extend([r'D:\dl_project\dl_project_cnn\DataInput', r'D:\dl_project\dl_project_cnn\Models' \
                    , r'D:\dl_project\dl_project_cnn\Trainer', r'D:\dl_project\dl_project_cnn\Utils'])

import configparser

from DataInput.Dataset import BasicDataset
from torch.utils.data import DataLoader

from torch import optim
from Utils.lr_scheduler import *

from Utils.metrics import *
from Utils.loss import *
from Utils.visualizer import *
from Utils.saver import *

import logging

logging.getLogger().setLevel(logging.INFO)
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp

import os
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed((seed))
    # torch.backends.cudnn.deterministics = True

def train(net,
          config_path
          ):
    '''record the start time'''
    start_time = time.clock()

    setup_seed(42)

    '''read the config file'''
    cf = configparser.ConfigParser()
    cf.read(config_path)
    secs = cf.sections()
    batch_size = int(cf.get(secs[0], 'batch_size'))
    root_path = cf.get(secs[0], 'root_path')
    lr_base = float(cf.get(secs[0], 'lr_base'))
    epochs = int(cf.get(secs[0], 'epochs'))
    interval_step = int(cf.get(secs[0], 'interval_step'))
    val_step = int(cf.get(secs[0], 'val_step'))
    dir_checkpoint = cf.get(secs[0], 'dir_checkpoint')

    '''read input data, using BasicDataset and Dataloader'''
    train_dataset = BasicDataset(root=root_path, image_set='train', transform=True)
    # train_dataset = BasicDataset(root=root_path, image_set='train', transform=False)
    val_dataset = BasicDataset(root=root_path, image_set='val')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)  # pred val_sample one by one
    max_iters = len(train_dataloader) * epochs

    '''appoint optimizer and lr_sched
    uler'''
    # optimizer = optim.SGD(net.parameters(), lr=lr_base, momentum=0.9, nesterov=False)
    optimizer = optim.AdamW(net.parameters(), lr=lr_base, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=5e-4, amsgrad=False)
    # optimizer = optim.NAdam(net.parameters(), lr=lr_base, betas=(0.9, 0.999), eps=1e-8,
    #              weight_decay=5e-4)
    lr_scheduler = ConsineAnnWithWarmup(optimizer=optimizer, loader_len=len(train_dataloader),  lr_max=lr_base, \
                                        lr_min=lr_base*0.1,  warm_prefix=True, epo_tot=60, epo_mult=1, warm_steps=1)
    # lr_scheduler = PolyLR(optimizer, base_lr=lr_base, max_iters=max_iters)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    '''set loss function and metrics class'''
    # criterion = nn.BCEWithLogitsLoss(torch.tensor([15.84]).cuda())
    # criterion = nn.MSELoss()
    # criterion =nn.BCEWithLogitsLoss()
    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_ori = nn.CrossEntropyLoss()
    criterion_edge = BCEDiceLoss()
    mtw = MultiTaskWrapper(task_num=3)
    # slaw = SLAW(num_tasks=3, loss_weights=[1., 1., 1.])
    # criterion_val = nn.BCEWithLogitsLoss()
    # pos_weight = torch.tensor([0.8]).cuda()
    # criterion_shp = nn.BCEWithLogitsLoss(torch.tensor([15.84]).cuda())
    # criterion = MSEDiceLoss(gamma=0.3)
    train_metricer1 = StreamSegMetrics(n_classes=2)
    train_metricer2 = StreamSegMetrics(n_classes=2)
    # train_metricer2_2 = StreamSegMetrics(n_classes=2)
    visualizer = Visualizer(comment=f'LR_{lr_base}_BS_{batch_size}_MAXiters_{max_iters}')

    '''monitoring iteration'''
    crt_iter = 0
    net.cuda()
    use_amp = False
    scaler = amp.GradScaler(enabled=use_amp)

    '''initialize the current best OA'''
    iou = 0
    saver = Saver(base_path=os.getcwd(), basis_name='IOU_')

    epo_cur = 0

    '''start training'''
    for epoch in range(epochs):
        '''load net to cuda'''

        net.train()
        logging.info(f'start training {epoch+1}/{epochs}.....')

        '''through train_loader monitoring the progress in each epoch'''
        loader = tqdm(train_dataloader, desc=f'training {epoch+1}/{epochs}', unit='batch', total=len(train_dataloader))

        for batch in loader:

            '''update current iteration'''
            crt_iter += 1
            # lr_scheduler.update(c_iters=crt_iter)

            '''get data and load to cuda'''
            inputs = batch['image'].cuda()  # BCHW
            labels = batch['label'].cuda()
            # dis_label = batch['dis_label'].cuda()
            o_label = batch['o_label'].cuda()
            edge_label = batch['edge_label'].cuda()
            vis_label = (torch.max(o_label.data, dim=1, keepdim=True).indices <=35).type(torch.float32)

            '''reset the optimizer'''
            optimizer.zero_grad()

            epo_cur += 1/len(train_dataloader)

            '''predict the output'''
            with amp.autocast(enabled=use_amp):
                outputs,  outputs2, outputs3= net(inputs) # seg ori edge
                # outputs2 = net(inputs)

                loss = criterion_seg(outputs, labels)
                loss2 = criterion_ori(outputs2, o_label)
                loss3 = criterion_edge(outputs3, edge_label)
                loss, loss2, loss3 = mtw([loss, loss2, loss3])
                loss_a = loss2 + loss + loss3
                # loss_a = loss

            # outputs = net(inputs)  # without sigmoid

            '''calculate the loss with l2'''

            outputs_prob = nn.Sigmoid()(outputs.data)
            named_values_indices = torch.max(nn.Softmax(dim=1)(outputs2.data), dim=1, keepdim=True)
            outputs_prob2 = named_values_indices.indices
            outputs_prob3 =  nn.Sigmoid()(outputs3.data)

            '''back propagation, and update the parameters'''
            scaler.scale(loss_a).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            # torch.nn.utils.clip_grad_value_(net.parameters(), 0.1)

            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            '''uodate the metric'''
            pred1 = (outputs_prob.data.cpu() > 0.5).type(torch.float32)
            pred2 = (outputs_prob2.data.cpu() <= 35).type(torch.float32)
            pred3 = (outputs_prob3.data.cpu() > 0.5).type(torch.float32)
            train_metricer1.update(labels.data.cpu().numpy(), pred1.data.cpu().numpy())
            train_metricer2.update(labels.data.cpu().numpy(), pred2.data.cpu().numpy())


            if crt_iter % interval_step == 0:
                '''monitoring on the cmd'''
                metric_dict1 = train_metricer1.get_results()
                metric_dict2 = train_metricer2.get_results()
                # train_metricer.reset()
                loader.set_description(f'Training {epoch+1}/{epochs} '
                                       f'Current loss1:{"%.2f" % loss.data} '
                                       f'Current loss2:{"%.2f" % loss2.data} '
                                       f'Precision:{"%.2f" % metric_dict1["Precision_fore"]} '
                                       f'Recall:{"%.2f" % metric_dict1["Recall_fore"]} '
                                       f'iou:{"%.2f" % metric_dict1["Class IoU"][1]} '
                                       f'iou2:{"%.2f" % metric_dict2["Class IoU"][1]} '
                                       )

                '''visualize, including lr, loss, acc'''
                visualizer.vis_scalar('lr', optimizer.param_groups[0]['lr'], crt_iter)
                visualizer.vis_scalar('train/loss1', loss.data, crt_iter)
                visualizer.vis_scalar('train/loss2', loss2.data, crt_iter)
                visualizer.vis_scalar('train/loss3', loss3.data, crt_iter)
                visualizer.vis_scalar('train/Precision', metric_dict1["Precision_fore"], crt_iter)
                visualizer.vis_scalar('train/Recall', metric_dict1["Recall_fore"], crt_iter)
                visualizer.vis_scalar('train/iou1', metric_dict1["Class IoU"][1], crt_iter)
                visualizer.vis_scalar('train/iou2', metric_dict2["Class IoU"][1], crt_iter)

                '''transfer images to cpu'''
                inputs_cpu = inputs.data.cpu()
                label_cpu = labels.data.cpu()
                # shape_label_cpu = shape_labels.data.cpu()
                pred1_cpu = pred1.data.cpu()
                pred2_cpu = pred2.data.cpu()
                pred3_cpu = pred3.data.cpu()

                '''visualize'''
                visualizer.vis_images(tag='train_vis/image', img_tensor=inputs_cpu, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/label', img_tensor=label_cpu, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/o_labels', img_tensor=vis_label, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/edge_label', img_tensor=edge_label, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/pred1', img_tensor=pred1_cpu, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/pred2', img_tensor=pred2_cpu, global_step=crt_iter)
                visualizer.vis_images(tag='train_vis/pred3', img_tensor=pred3_cpu, global_step=crt_iter)

            if crt_iter % val_step == 0 or crt_iter == 1:

                '''validate the trained model'''
                loss_val, val_acc_dict, val_acc_dict2, val_acc_dict3= validate(net=net, val_loader=val_dataloader,
                                                  criterion=criterion_seg, visualizer=visualizer, global_step=crt_iter)
                # loss_val, val_acc_dict= validate(net=net, val_loader=val_dataloader,
                #                                   criterion=criterion_seg, visualizer=visualizer, global_step=crt_iter)
                loader.set_description(f'validation {epoch+1}/{epochs} '
                                       f'loss:{"%.2f" % loss_val.data} '
                                       f'Precision:{"%.2f" % val_acc_dict["Precision_fore"]} '
                                       f'Recall:{"%.2f" % val_acc_dict["Recall_fore"]} '
                                       f'iou:{"%.2f" % val_acc_dict["Class IoU"][1]} '
                                       f'iou2:{"%.2f" % val_acc_dict2["Class IoU"][1]} '
                                       f'iou3:{"%.2f" % val_acc_dict3["Class IoU"][1]} '
                                       )

                '''resave the current best model'''
                res_list = [val_acc_dict["Class IoU"][1], val_acc_dict2["Class IoU"][1], val_acc_dict3["Class IoU"][1]]
                acc_dict = [val_acc_dict, val_acc_dict2, val_acc_dict3]
                branch_list = ['_SEGMIOU', '_ORIMIOU', '_CPRMIOU']
                best_iou = max(res_list)
                best_ind = res_list.index(best_iou)
                if best_iou>=iou:
                    saver.update(model=net, value=str(best_iou) + branch_list[best_ind] + str(acc_dict[best_ind]["Mean IoU"]))
                    iou = best_iou
                # if val_acc_dict["Class IoU"][1] >= iou:
                #         saver.update(model=net, value=str(val_acc_dict["Class IoU"][1])+'_SEGMIOU'+str(val_acc_dict["Mean IoU"]))
                #         iou = val_acc_dict["Class IoU"][1]
                '''visualize, including lr, loss, acc'''
                visualizer.vis_scalar('val/loss', loss_val.data, crt_iter)
                visualizer.vis_scalar('val/Precision1', val_acc_dict["Precision_fore"], crt_iter)
                visualizer.vis_scalar('val/Recall1', val_acc_dict["Recall_fore"], crt_iter)
                visualizer.vis_scalar('val/iou', val_acc_dict["Class IoU"][1], crt_iter)
                visualizer.vis_scalar('val/Precision2', val_acc_dict2["Precision_fore"], crt_iter)
                visualizer.vis_scalar('val/Recall2', val_acc_dict2["Recall_fore"], crt_iter)
                visualizer.vis_scalar('val/iou_o', val_acc_dict2["Class IoU"][1], crt_iter)
                visualizer.vis_scalar('val/iou_a', val_acc_dict3["Class IoU"][1], crt_iter)

            lr_scheduler.step()

        '''after training a epoch completely, calculate the whole acc, save a checkpoint'''
        final_acc_dic = train_metricer1.get_results()
        if epoch%1==0:
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}MIOU{final_acc_dic["Mean IoU"]}.pth')  # 以字典形式保存了模型的所有参数
        logging.info(f'Checkpoint {epoch + 1} saved !')
        '''reset the train metric'''
        train_metricer1.reset()

        '''complete a step (epoch) for lr_scheduler'''
        # lr_scheduler.step()

    '''complete all epochs, calculate and output running time'''
    end_time = time.clock()
    running_time = (end_time - start_time) / 3600
    logging.info(f'Running time: {running_time}hours')


def validate(net, val_loader, criterion, visualizer, global_step):
    '''change the network to validation mode'''
    net.eval()

    '''initialize the loss and metric'''
    val_metricer = StreamSegMetrics(n_classes=2)
    val_metricer2 = StreamSegMetrics(n_classes=2)
    val_metricer3 = StreamSegMetrics(n_classes=2)

    loss = 0

    '''create the iterater'''
    loader = tqdm(val_loader, desc=r'validating', unit='img', total=len(val_loader))
    visual_idx = np.random.randint(0, len(loader))

    for i, data in enumerate(loader):

        input = data['image'].cuda()
        label = data['label'].cuda()

        with torch.no_grad():
            output, output2, output3 = net(input)
            # output = net(input)

            output_prob = nn.Sigmoid()(output.data.cpu())
            output2_prob = (torch.max(nn.Softmax(dim=1)(output2.data.cpu()), dim=1, keepdim=True).indices<= 35).type(torch.float32)
            output3_prob =nn.Sigmoid()(output3.data.cpu())
            output_prob_a = (output_prob + output2_prob*0.9+0.1)/2  # 方向分支+分割分支

            pred = (output_prob.data > 0.5).type(torch.float32)
            pred2 = output2_prob.data
            pred3 = (output3_prob > 0.5).type(torch.float32)
            preda = (output_prob_a.data > 0.5).type(torch.float32)

            '''update loss and metric'''
            loss = criterion(output, label) + loss
            val_metricer.update(label.data.cpu().numpy(), pred.data.cpu().numpy())
            val_metricer2.update(label.data.cpu().numpy(), pred2.data.cpu().numpy())
            val_metricer3.update(label.data.cpu().numpy(), preda.data.cpu().numpy())

            if i == visual_idx:

                '''transfer images to cpu'''
                val_input_cpu = input.data.cpu()
                val_label_cpu = label.data.cpu()
                val_pred_cpu = pred.data.cpu()
                val_pred_cpu2 = pred2.data.cpu()
                val_pred_cpu3 = pred3.data.cpu()
                val_pred_cpua = preda.data.cpu()


                '''visualize'''
                visualizer.vis_images(tag='val_vis/image', img_tensor=val_input_cpu, global_step=global_step)
                visualizer.vis_images(tag='val_vis/label', img_tensor=val_label_cpu, global_step=global_step)
                visualizer.vis_images(tag='val_vis/pred', img_tensor=val_pred_cpu, global_step=global_step)
                visualizer.vis_images(tag='val_vis/pred2', img_tensor=val_pred_cpu2, global_step=global_step)
                visualizer.vis_images(tag='val_vis/pred3', img_tensor=val_pred_cpu3, global_step=global_step)
                visualizer.vis_images(tag='val_vis/preda', img_tensor=val_pred_cpua, global_step=global_step)

    res_dict = val_metricer.get_results()
    res_dict2 = val_metricer2.get_results()
    res_dict3 = val_metricer3.get_results()

    loss = loss / len(val_loader)

    '''change back to training'''
    net.train()

    return loss, res_dict, res_dict2, res_dict3
    # return loss, res_dict



