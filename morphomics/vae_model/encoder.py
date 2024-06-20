import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dimensions=[16, 8]):
        super(Encoder, self).__init__()
        
        # Define the layer dimensions
        # time 2 laten dim because we need the mean and the variance of the dist.
        self.layer_dimensions = [input_dim] + hidden_dimensions + [2*latent_dim]
        self.z_dist_dim = int(self.layer_dimensions[-1]/2)

        self.num_layers = len(self.layer_dimensions) - 1 
        # Initialize an empty list to hold the layers
        layers = []

        # Loop through the layer dimensions to create Linear and ReLU layers
        for i in range(len(self.layer_dimensions) - 1):
            # Add a Linear layer
            layers.append(torch.nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
            
            # Add a ReLU layer, except after the last Linear layer
            if i < len(self.layer_dimensions) - 2:
                layers.append(torch.nn.ReLU())
        print(layers)
        # Create the Sequential model with the layers
        self.model = torch.nn.Sequential(*layers)
        

    def forward(self, x):
        # Pass the input through the model
        z_dist_params = self.model(x)
        z_mean, z_log_var = z_dist_params[:, :self.z_dist_dim], z_dist_params[:, self.z_dist_dim:] 
        return  z_mean, z_log_var