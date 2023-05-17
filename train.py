from tqdm import tqdm
from utils.utils import mIOU
import torch
import os
import matplotlib.pyplot as plt


def train(model, device, train_loader, optimizer, loss_function):
    running_loss = 0.0
    running_mIOU = 0.0

    for batch in tqdm(train_loader):
        # image, labels, _, _ = batch
        image, labels,_,_ = batch
        # print(torch.unique(labels))
        # print(torch.unique(labels))
        image, labels = image.to(device), labels.to(device)
    #     # # print(torch.max(labels), torch.min(labels))
        
        prediction = model(image)
        # print('prediction shape: ',prediction.shape)
        # print('label shape: ', labels.shape)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(prediction.cpu().detach().numpy()[0][0])  # Assuming data is in channel-last format

        # # Plot the corresponding label mask
        # plt.subplot(1, 2, 2)
        # plt.imshow(labels[0].cpu().detach().numpy())
        # plt.show()
        optimizer.zero_grad()
        loss = 0.8*loss_function(prediction, labels)  - 0.2*mIOU(labels, prediction)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*image.size(0)
        running_mIOU += mIOU(labels, prediction)

    # calculate average loss
    running_loss = running_loss/len(train_loader)
    running_mIOU = running_mIOU/len(train_loader)
    return running_loss, running_mIOU


  
def evaluate(model, data_loader, device, loss_function):
    running_loss = 0.0
    running_mIOU = 0.0
    with torch.no_grad():
        model.eval()
        for image, labels, _, _ in data_loader:
            image, labels = image.to(device), labels.to(device)
            prediction = model(image)
            loss = 0.8*loss_function(prediction, labels)  - 0.2*mIOU(labels, prediction) 
            running_loss += loss.item()*image.size(0)
            running_mIOU += mIOU(labels, prediction)
        running_loss = running_loss/len(data_loader)
        running_mIOU = running_mIOU/len(data_loader)

    return running_loss, running_mIOU

def train_model(num_epochs, model, device, train_loader, optimizer, loss_function,  save_path, scheduler = None, val_loader = None):
    print("Start training...")
    model = model.to(device)
    for epoch in range(1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print("Starting Epoch "+str(epoch))
        train_loss, running_mIOU = train(model, device, train_loader, optimizer, loss_function)
        val_loss, val_mIOU = evaluate(model, val_loader, device, loss_function)
        if scheduler is not None:

            scheduler.step(val_loss)
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train IOU: {:.4f}, Val Loss: {:.4f}, Val IOU: {:.4f}'.format(epoch, num_epochs, train_loss, running_mIOU, val_loss, val_mIOU))
        if epoch%10 == 0:
            save_checkpoint(save_path=save_path, model=model, optimizer=optimizer, val_loss=0, epoch=epoch)

def save_checkpoint(save_path, model, optimizer, val_loss, epoch):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    file_name = save_path.split("/")[-1].split("_")[0] + "_" + str(epoch) + ".pt"
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, os.path.join(save_path, file_name))


    