import torch 
import torch_geometric 

def vae_train(data, model, sample_size, optimizer, loss_fn, epochs, batch_size,
          feature_scaler = None, sample_scaler = None):
    
    if sample_scaler:
        data = sample_scaler.fit_transform(data)

    if feature_scaler:
        data = feature_scaler.fit_transform(data)
    
    # Create a DataLoader
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
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
            out, z_mean, z_log_var = model(x, sample_size = sample_size)
            
            # Calculate the loss
            loss, mse = loss_fn(x, out, z_mean, z_log_var)
            
            # Backpropagate
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Add the loss to the total loss
            tot_loss += loss.item()
            tot_mse += mse
        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {tot_loss/(i+1)}, mse: {tot_mse/(i+1)}')
    
    return model


def vae_test(data, model, sample_size, loss_fn = None,
         sample_scaler = None, feature_scaler = None):

    if sample_scaler:
        data = sample_scaler.transform(data)

    if feature_scaler:
        data = feature_scaler.transform(data)
    
    model.eval()
    with torch.no_grad():
        # Pass the data through the model
        out, z_mean, z_log_var = model(data, sample_size = sample_size)

        # compute mse
        x_expanded = data.unsqueeze(0).expand(*out.shape)
        l2 = torch.norm(out - x_expanded, dim = -1, p=2)
        mse = torch.mean(l2)

    return out, z_mean, z_log_var, mse


def gnn_train(data, model, optimizer, loss_fn, epochs, batch_size,
          feature_scaler = None, sample_scaler = None):
    
    if sample_scaler:
        data = sample_scaler.fit_transform(data)

    if feature_scaler:
        data = feature_scaler.fit_transform(data)
    
    # Create a DataLoader
    loader = torch_geometric.loader.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    # Loop through the epochs
    for epoch in range(epochs):
        # Initialize the loss
        tot_loss = 0
        tot_acc = 0
        # Loop through the DataLoader
        for i, x in enumerate(loader):
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
        # Print the loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {tot_loss/(i+1)}, acc: {tot_acc/(i+1)}')
    
    return model


def gnn_test(data, model, loss_fn = None,
         sample_scaler = None, feature_scaler = None):

    if sample_scaler:
        data = sample_scaler.transform(data)

    if feature_scaler:
        data = feature_scaler.transform(data)
    
    model.eval()
    with torch.no_grad():
        # Pass the data through the model
        out = model(data)

        # compute mse
        loss, acc = loss_fn(data, out)

    return out, loss, acc