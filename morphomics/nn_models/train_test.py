import torch 
import torch_geometric 
from tqdm import tqdm
import numpy as np
from .criterion import VAELoss
def vae_train(data, model, sample_size, optimizer, loss_fn, epochs, batch_size, scheduler=None):
    # If it is an image add a channel dim
    if len(data.shape) == 3:
        data = data.unsqueeze(1)  # Adds a channel dimension, so the shape becomes (batch_size, 1, 100, 100)

    # Create a DataLoader with shuffling enabled (it's already shuffled each epoch)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # Create kl factors
    x_values = np.linspace(2, 7, epochs)  # Generate 100 points between 0 and 5
    kl_factor_list = (1 - np.exp(-x_values))

    # Loop through the epochs
    for epoch in range(epochs):
        # Initialize the loss
        tot_loss = 0
        tot_mse = 0
        # Loop through the DataLoader
        for i, x in enumerate(loader):
            # Zero the gradients
            optimizer.zero_grad()

            # Pass the input through the model
            out, z_mean, z_log_var = model(x, sample_size=sample_size)
            # Calculate the loss
            loss, mse = loss_fn.forward(x, out, z_mean, z_log_var, kl_factor_list[epoch])
            
            # Backpropagate
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Add the loss to the total loss
            tot_loss += loss.item()
            tot_mse += mse

        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {tot_loss/(i+1)}, mse: {tot_mse/(i+1)}, i: {i}')
            if scheduler:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}, Current Learning Rate: {current_lr}')
        
        # Step the learning rate scheduler if provided
        if scheduler:
            scheduler.step(tot_loss / (i+1))  # Optional: you can use the loss or any metric for stepping

    return model

def vae_test(data, model, sample_size):
    # If it is an image add a channel dim
    if len(data.shape) == 3:
        data = data.unsqueeze(1)  # Adds a channel dimension, so the shape becomes (batch_size, 1, 100, 100)
    criterion = VAELoss()
    criterion.eval()
    model.eval()
    with torch.no_grad():
        # Pass the data through the model
        out, z_mean, z_log_var = model(data, sample_size = sample_size)
        # compute mse
        mse = criterion.get_mse(data, out)

    return out, z_mean, z_log_var, mse


def gnn_train(train_data, test_data, model, optimizer, loss_fn, epochs, batch_size, print_every = 10):
    
    # Create DataLoaders
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    
    # Initialize trackers
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Loop through the epochs
    for epoch in range(epochs):
        # Initialize the loss
        tot_loss = 0
        tot_acc = 0
        # Loop through the train DataLoader
        model.train()
        for i, x in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Zero the gradients
            optimizer.zero_grad()

            # Pass the input through the model
            out = model(x)
            
            # Calculate the loss
            loss, acc = loss_fn(x, out)
            
            # Backpropagate
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Add the loss to the total loss
            tot_loss += loss.item()
            tot_acc += acc
        
        # Calculate average loss and accuracy for the epoch
        avg_train_loss = tot_loss / (i+1)
        avg_train_acc = tot_acc / (i+1)
        
        # Append to trackers
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # Evaluate on test set every epoch
        _, test_loss, test_acc = gnn_test(next(iter(test_loader)), model, loss_fn)
        
        # Append to trackers
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print the loss every 10 epochs
        if epoch % print_every == 0:
            print(f'Epoch {(epoch+1):03d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
    
    return model, train_losses, train_accs, test_losses, test_accs


def gnn_test(data, model, loss_fn = None):
    
    model.eval()
    with torch.no_grad():
        # Pass the data through the model
        out = model(data)

        # compute loss and accuracy
        loss, acc = loss_fn(data, out)

    return out, loss, acc