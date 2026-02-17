import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check allow for CUDA usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(nn.Module):
    """
     Convolutional LSTM Cell for Spatiotemporal Data processing.
    
    Args:
        input_dim (int): Number of channels in input tensor.
        hidden_dim (int): Number of channels in hidden state.
        kernel_size (tuple): Size of the convolutional kernel.
        bias (bool): Whether or not to add the bias.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class HeatwaveConvLSTM(nn.Module):
    """
    Encoder-Decoder ConvLSTM Model for Sequence-to-Sequence Prediction.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(HeatwaveConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=True))

        self.cell_list = nn.ModuleList(cell_list)

        # Final convolution to map hidden state to output channels
        # We want to predict the next frame, so output channels = input channels
        self.final_conv = nn.Conv2d(self.hidden_dim[-1], input_dim, kernel_size=(1, 1))


    def forward(self, input_tensor, future_seq=1):
        """
        Args:
            input_tensor: (Batch, Time, Channels, Height, Width)
            future_seq: Number of future time steps to predict.
        """
        b, t, c, h, w = input_tensor.size()
        
        # Initialize hidden states
        hidden_state = []
        for i in range(self.num_layers):
            hidden_state.append(self.cell_list[i].init_hidden(b, (h, w)))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)

        # Encoder: Process input sequence
        for t in range(seq_len):
            current_input = input_tensor[:, t, :, :, :]
            
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                h, c = self.cell_list[layer_idx](current_input, (h, c))
                hidden_state[layer_idx] = (h, c)
                current_input = h # Output of current layer is input to next layer

        # Decoder: Predict future sequence
        # For simplicity in this example, we'll use the last hidden state to predict the next frame
        # In a more complex Seq2Seq, we might feed the output back as input
        
        # We will iterate for future_seq steps. 
        # The input to the first decoder step is the last input frame (or zero, or output of encoder).
        # Here we use the last generated hidden state to predict the next frame.
        
        outputs = []
        
        # Start with the last input frame
        current_input = input_tensor[:, -1, :, :, :]
        
        for t in range(future_seq):
             # Run through layers
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                # We feed the output of the previous layer as input to the next
                # But for the first layer, what is the input? 
                # Standard Seq2Seq: feed the *prediction* from the previous time step.
                # Here, let's assume we are just unrolling the LSTM states.
                
                # However, ConvLSTM Cells expect an input. 
                # Strategy: Feed the output of the last timestep of the encoder as the first input to decoder? 
                # Or use a zero tensor? Let's use the prediction from the previous step.
                
                # First step of decoder:
                if layer_idx == 0:
                     # For the very first layer in decoder, we can feed the last known frame 
                     # or the prediction from the previous step.
                     pass
                
                h, c = self.cell_list[layer_idx](current_input, (h, c))
                hidden_state[layer_idx] = (h, c)
                current_input = h

            # Map final hidden state to output
            prediction = self.final_conv(current_input)
            outputs.append(prediction.unsqueeze(1))
            
            # Use specific prediction as input for next step (Autoregressive)
            current_input = prediction 

        outputs = torch.cat(outputs, dim=1)
        return outputs

class PhysicsInformedLoss(nn.Module):
    """
    Custom Loss component combining MSE with Physics-Informed constraints.
    Constraint: Adiabatic Compression/Expansion.
    Rationale:
        - Sinking air (Subsidence) -> Warming (Adiabatic Compression).
        - Rising air (Ascent) -> Cooling (Adiabatic Expansion).
        - In weather terms: 
            - Increasing Geopotential Height (Ridge building, High Pressure) ~ Subsidence ~ Warming.
            - Decreasing Geopotential Height (Trough deepening, Low Pressure) ~ Ascent ~ Cooling.
        - Therefore, dZ/dt and dT/dt should be positively correlated.
    """
    def __init__(self, lambda_phy=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy

    def forward(self, prediction, target):
        """
        Args:
            prediction: (Batch, Time, Channels, Height, Width)
            target: (Batch, Time, Channels, Height, Width)
            Channels assumed: 0: Z500, 1: T2m, 2: Soil Moisture
        """
        # 1. Data Loss (MSE)
        mse_loss = self.mse(prediction, target)

        # 2. Physics Loss (Thermodynamic Consistency)
        # We need to approximate time derivatives dZ/dt and dT/dt
        # We can use finite differences along the time dimension.
        
        # Slice channels
        z500_pred = prediction[:, :, 0, :, :]
        t2m_pred = prediction[:, :, 1, :, :]
        
        # Calculate time derivatives (d/dt)
        # dz/dt approx z(t+1) - z(t)
        dz_dt = z500_pred[:, 1:, :, :] - z500_pred[:, :-1, :, :]
        dt_dt = t2m_pred[:, 1:, :, :] - t2m_pred[:, :-1, :, :]

        # Physics Constraint:
        # We want dz_dt and dt_dt to have the same sign (positive correlation).
        # If they have different signs, their product will be negative.
        # We want to penalize negative products.
        # However, it's easier to enforce that correlation > 0 or that the product is positive.
        # Let's penalize where sign(dz_dt) != sign(dt_dt).
        # Soft penalty: ReLU( - (dz_dt * dt_dt) )
        # If product is positive (consistent), -product is negative, ReLU is 0.
        # If product is negative (inconsistent), -product is positive, ReLU returns the magnitude of specific inconsistency.
        
        physics_violation = torch.relu( - (dz_dt * dt_dt) )
        physics_loss = torch.mean(physics_violation)

        total_loss = mse_loss + self.lambda_phy * physics_loss

        return total_loss, mse_loss, physics_loss

# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 2
    TIME_STEPS = 5
    CHANNELS = 3 # Z500, T2m, Soil Moisture
    HEIGHT = 64
    WIDTH = 64
    HIDDEN_DIM = [16, 16]
    KERNEL_SIZE = [(3, 3), (3, 3)]
    NUM_LAYERS = 2
    FUTURE_SEQ = 2

    # Initialize Model
    model = HeatwaveConvLSTM(input_dim=CHANNELS,
                             hidden_dim=HIDDEN_DIM,
                             kernel_size=KERNEL_SIZE,
                             num_layers=NUM_LAYERS).to(device)

    # Initialize Loss
    criterion = PhysicsInformedLoss(lambda_phy=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Generate Dummy Data
    # (B, T, C, H, W)
    input_seq = torch.randn(BATCH_SIZE, TIME_STEPS, CHANNELS, HEIGHT, WIDTH).to(device)
    target_seq = torch.randn(BATCH_SIZE, FUTURE_SEQ, CHANNELS, HEIGHT, WIDTH).to(device)

    print("Model Architecture:")
    print(model)

    # Training Step
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    # We predict the future sequence based on the input sequence
    output_seq = model(input_seq, future_seq=FUTURE_SEQ)

    # Calculate Loss
    loss, mse, phy = criterion(output_seq, target_seq)

    # Backward Pass
    loss.backward()
    optimizer.step()

    print("\n--- Training Step Output ---")
    print(f"Input Shape: {input_seq.shape}")
    print(f"Target Shape: {target_seq.shape}")
    print(f"Output Shape: {output_seq.shape}")
    print(f"Total Loss: {loss.item():.6f}")
    print(f"MSE Loss: {mse.item():.6f}")
    print(f"Physics Loss: {phy.item():.6f}")
    print("Gradients propagated successfully.")
