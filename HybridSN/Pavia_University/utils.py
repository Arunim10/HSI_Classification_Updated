import torch
import numpy as np

### Evaluation Metrics
from sklearn.metrics import f1_score
def f_score(y_true,y_pred):
    ## labels = [0,5,2,1,4]
    ## predictions = [0,5,1,4,2]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = f1_score(y_true, y_pred, average=None)
    return scores
def accuracy(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true==y_pred).sum()/len(y_true)
    return acc

### Evaluation Function
from tqdm import tqdm
def evaluate_model(model,dl,device,name):
    torch.manual_seed(42)
    model = model.to(device)
    pred = []
    with torch.no_grad():
        model.eval()
        losses = []
        targets = []
        with tqdm(dl,desc=f'{name}',mininterval=30) as progress:
            for i,(spec_values,labels) in enumerate(progress):
                    spec_values = spec_values.to(device)
                    labels = labels.to(device,dtype=torch.int64)
                    label_pred = model(spec_values.to(torch.float32))
                    loss = torch.nn.functional.cross_entropy(label_pred,labels).item()
                    losses.append(loss)
                    lab_out = torch.argmax(label_pred,dim=1)
                    pred.append(lab_out.cpu().numpy())
                    targets.append(labels.cpu().numpy())
                    f1_temp = f_score(labels.cpu().numpy(),lab_out.cpu().numpy())
                    acc_temp = accuracy(labels.cpu().numpy(),lab_out.cpu().numpy())
                    progress.set_description(f'loss:{loss:.4f},f_sc:{np.mean(f1_temp):.4f},acc:{acc_temp:.4f}')
    pred = np.concatenate(pred)
    targets = np.concatenate(targets)
    f1_sc = f_score(targets,pred)
    acc = accuracy(targets,pred)
    return pred,targets,losses,f1_sc,acc

### Training Function
import gc
def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
def train_model(model,train_dl,test_dl,epochs,device,lr,wd):
    torch.manual_seed(42)
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=1e-6)
    best_eval_score = 0
    val_loss = []
    train_loss = []
    val_fscore = []
    train_fscore = []
    for epoch in range(epochs):
        print("Epoch ",epoch)
        model.train()
        with tqdm(train_dl,desc='Train',mininterval=30) as train_progress:
            for i,(spec_values,labels) in enumerate(train_progress):
                    optim.zero_grad()
                    spec_values = spec_values.to(device)
                    labels = labels.to(device,dtype=torch.int64)
                    label_pred = model(spec_values.to(torch.float32))
                    # print(label_pred)
                    # print(labels)
                    loss = torch.nn.functional.cross_entropy(label_pred,labels)
                    lab_out = torch.argmax(label_pred,dim=1)
                    f1_temp = f_score(labels.cpu().numpy(),lab_out.cpu().numpy())
                    acc_temp = accuracy(labels.cpu().numpy(),lab_out.cpu().numpy())
                    if (np.isinf(loss.item()) or np.isnan(loss.item())):
                        print(f'Bad loss, skipping the batch {i}')
                        del loss,label_pred
                        gc_collect()
                        continue
                        
                    ## Training Code    
                    loss.backward()
                    optim.step()
                    scheduler.step(loss)
                    
                    lr = scheduler.get_last_lr()[0] if scheduler else lr
                    train_progress.set_description(f'loss:{loss:.4f},lr:{lr:.4f},f1:{np.mean(f1_temp):.4f},acc:{acc_temp:.4f}')
                    
        if test_dl is not None:
            pred_val,targets_val,losses_val,f1_val,acc_val = evaluate_model(model,test_dl,device,'Eval')
            pred_train,targets_train,losses_train,f1_train,acc_train = evaluate_model(model,train_dl,device,'Train_Eval')
            val_loss.append(np.mean(losses_val))
            train_loss.append(np.mean(losses_train))
            val_fscore.append(np.mean(f1_val))
            train_fscore.append(np.mean(f1_train))
            if (np.mean(f1_val)>best_eval_score):
                print(best_eval_score," ---> ",np.mean(f1_val))
                best_eval_score = np.mean(f1_val)
                
            print("Per Epoch Test f1: ",f1_val)
            print("Per Epoch Accuracy: ",acc_val)
                
    if test_dl is not None:
        pred,targets,losses,f1,acc = evaluate_model(model,test_dl,device,'Eval')
        
        print("Final Test f1: ",f1)
        print("Final Accuracy: ",acc)
    return model,train_loss,val_loss,train_fscore,val_fscore