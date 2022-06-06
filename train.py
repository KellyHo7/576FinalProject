import os
import torch
import helper
import argparse
import numpy as np
import torch.optim as optim
from util.Mydataset import *
from util.DHG_parse_data import *
#from util.SHREC_parse_data import *
from model.DG_STA_model import *



def init_data_loader(test_subject_id, data_cfg, batch_size, workers):

    train_data, test_data = get_train_test_data(test_subject_id, data_cfg)
    #train_data, test_data = split_train_test(data_cfg)

    train_dataset = Hand_Dataset(train_data, use_data_aug = True, time_len = 8)
    test_dataset = Hand_Dataset(test_data, use_data_aug = False, time_len = 8)

    """
    print("train data num: ",len(train_dataset))
    print("test data num: ",len(test_dataset))

    print("batch size:", args.batch_size)
    print("workers:", args.workers)
    """

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers)

    validloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=workers)

    return trainloader, validloader



def init_model(data_cfg, dp_rate):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    model = DG_STA(class_num, dp_rate)
    # Run operations on multiple GPUs by making the model run parallellly
    model = torch.nn.DataParallel(model) #.cuda() 

    return model


def run(sample_batched,model,criterion):
    data = sample_batched["skeleton"].float()
    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = torch.autograd.Variable(label, requires_grad=False)

    score = model(data)
    loss = criterion(score,label)
    
    output = score.cpu().data.numpy()
    label = label.cpu().data.numpy()
    output = np.argmax(output, axis=1)
    acc = np.sum(output==label)/float(label.size)

    return score,loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DT_STA_model')
    
    parser.add_argument('-b', '--batch_size', type=int, default=32,help='mini-batch size') 
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--cuda', default=True, help='enables cuda')
    #parser.add_argument('-c', '--use-cuda', type=str2bool, default=True, help='enables cuda')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--dp_rate', type=float, default=0.2, help='dropout rate')

    # The paper did not specify how many epochs used
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    #parser.add_argument('--patiences', default=50, type=int, help='number of epochs to tolerate no improvement of val_loss')
    
    parser.add_argument('--test_subject_id', type=int, default=3, help='id of test subject, for cross-validation')
    parser.add_argument('--data_cfg', type=int, default=0, help='0 for 14 class, 1 for 28')
    

    args = parser.parse_args()
    

    print("\nHyperparamter:")
    print(args)
    

    #folder for saving trained model
    model_fold = "./model/DHS_ID-{}_dp-{}_lr-{}_dc-{}/".format(args.test_subject_id,args.dp_rate, args.learning_rate, args.data_cfg)
    #model_fold = "./model/SHREC_dp-{}_lr-{}_dc-{}/".format(args.dp_rate, args.learning_rate, args.data_cfg)
    try:
        os.mkdir(model_fold)
    except:
        pass

    torch.manual_seed(1)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    trainloader, validloader = init_data_loader(args.test_subject_id, args.data_cfg, args.batch_size, args.workers)

    model = init_model(args.data_cfg, args.dp_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    cuda = torch.cuda.is_available() and args.cuda
    
    if cuda:
        model = model.cuda()
        
    max_acc = 0
    
    for epoch in range(args.epochs):
        train_acc = 0
        train_loss = 0
        for i, sample_batched in enumerate(trainloader):
            score,loss, acc = run(sample_batched, model, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += acc
            train_loss += loss

        train_acc /= float(i + 1)
        train_loss /= float(i + 1)

        print("DHS  Epoch: [%2d] "
              "cls_loss: %.4f  train_ACC: %.6f "
              % (epoch + 1, train_loss.data, train_acc))

        with torch.no_grad():
            val_acc = 0
            val_loss = 0
            for i, sample_batched in enumerate(validloader):         
                score, loss, acc = run(sample_batched, model, criterion)
                val_acc += acc
                val_loss += loss

            val_loss = val_loss / float(i + 1)
            val_acc = val_acc / float(i + 1)

            print("DHS  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f "
                  % (epoch + 1, val_loss, val_acc))


            #save best model and stop once reach 95% accuracy
            if val_acc > max_acc:
                max_acc = val_acc
                no_improve_epoch = 0
                val_acc = round(val_acc, 10)

                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_acc))
            """
            if val_acc > 0.95:
                print("stop training....")
                break
            """
