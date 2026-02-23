import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys,time
from os.path import join
import torch
from lib.losses.loss import *
from lib.common import *
from config import parse_args
from lib.logger import Logger, Print_Logger
import models
from test import Test
# from models.networks.neuron_net import *
from function import get_dataloader, train,train_thickness, val, get_dataloaderV2,get_dataloaderV3
# from models.UNetWave import U_Net_Wave
# import wandb
def main():
    setpu_seed(2021)
    args = parse_args()

    # config_wandb = wandb.config  # Initialize config
    # config_wandb.batch_size = args.batch_size  # input batch size for training (default:64)

    data_name = args.train_data_path_list.split('/')[-2]
    # print(args.thickness)
    save_path = join(args.outf, args.save)+'_'+data_name+'/'+args.model_name+'/'+args.pooling_type+'/'
    save_args(args,save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    
    log = Logger(save_path)
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')
    
    # net = models.UNetFamily.U_Net(1,2).to(device)
    if args.model_name == 'unet':
        net = models.U_Net(img_ch=args.in_channels, output_ch=args.classes,pooling_type=args.pooling_type).to(device)
    elif args.model_name == 'waveunet':
        net = models.U_Net_Wave_V2(img_ch = args.in_channels, output_ch = args.classes, wavename = 'haar').to(device)
    elif args.model_name == 'laddernet':
        net = models.LadderNet(inplanes=args.in_channels, num_classes=args.classes, layers=3, filters=16).to(device)
    print("Total number of parameters: " + str(count_parameters(net)))

    log.save_graph(net,torch.randn((1,1,48,48)).to(device).to(device=device))  # Save the model structure to the tensorboard file
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    if args.thickness == 1:
        ce_perpixel_loss = CrossEntropyLoss2d(reduce=False) # Initialize loss function
        ce_loss = CrossEntropyLoss2d()
        dice_loss = DiceLoss(2)
    else:
        ce_loss = CrossEntropyLoss2d()
        dice_loss = DiceLoss(2)
    criterion_val = CrossEntropyLoss2d()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # create a list of learning rate with epochs
    # lr_schedule = make_lr_schedule(np.array([50, args.N_epochs]),np.array([0.001, 0.0001]))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    if args.thickness == 1:
        train_loader, val_loader = get_dataloaderV3(args)
    else:
        train_loader, val_loader = get_dataloaderV2(args) # create dataloader
    # train_loader, val_loader = get_dataloader(args)
    
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args) 

    best = {'epoch':0,'AUC_roc':0.5,'val_f1':0} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        if args.thickness == 1:
            train_log = train_thickness(train_loader,net,ce_perpixel_loss,ce_loss,dice_loss, optimizer,device,epoch)
        else:
            train_log = train(train_loader,net,ce_loss,dice_loss, optimizer,device,epoch)  #todo

        # val stage
        if not args.val_on_test:
            val_log = val(val_loader,net,criterion_val,device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()
        # wandb.log({'test_f1': val_log['val_f1']})
        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_f1'] > best['val_f1']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['val_f1'] = val_log['val_f1']
            trigger = 0
        print('Best performance at Epoch: {} | val_f1: {}'.format(best['epoch'],best['val_f1']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    # wandb.init(project="VesselSeg",name='unet-drop-dice-thick')
    main()
