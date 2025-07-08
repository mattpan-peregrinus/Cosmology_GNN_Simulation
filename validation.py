import torch
import torch_geometric as pyg
from data_utils import preprocess

def momentum_conservation_loss(accelerations, dt, momentum_weight):
    velocity_changes = accelerations * dt
    total_momentum_change = torch.sum(velocity_changes, dim=0)
    momentum_loss = momentum_weight * torch.sum(total_momentum_change ** 2)
    return momentum_loss

def validate(model, val_loader, device, loss_fn, acc_loss_weight, temp_rate_loss_weight, momentum_loss_weight, metadata, noise_std, num_neighbors, dt, box_size):
    model.eval()  
    total_loss = 0.0
    acc_loss_total = 0.0
    temp_rate_loss_total = 0.0
    momentum_loss_total = 0.0
    count = 0
    
    with torch.no_grad():  
        for batch in val_loader:
            for key in batch["input"]:
                batch["input"][key] = batch["input"][key].float()
            for key in batch["target"]:
                batch["target"][key] = batch["target"][key].float()
            
            graphs = []
            for i in range(len(batch["input"]["Coordinates"])):
                input_coords = batch["input"]["Coordinates"][i]
                target_coords = batch["target"]["Coordinates"][i]
                input_temperature = batch["input"]["InternalEnergy"][i]
                target_temperature = batch["target"]["InternalEnergy"][i]
                
                graph = preprocess(
                    position_seq=input_coords,
                    target_position=target_coords,
                    temperature_seq=input_temperature,
                    target_temperature=target_temperature,
                    metadata=metadata,
                    noise_std=noise_std,
                    num_neighbors=num_neighbors,
                    dt=dt,
                    box_size=box_size
                )
                graphs.append(graph)
            
            if not graphs:
                continue
                
            batch_graph = pyg.data.Batch.from_data_list(graphs).to(device)
            
            # Forward pass
            predictions = model(batch_graph)
            acc_pred = predictions['acceleration']
            temp_rate_pred = predictions['temp_rate']
            
            acc_loss = loss_fn(acc_pred, batch_graph.y_acc)
            temp_rate_loss = loss_fn(temp_rate_pred, batch_graph.y_temp_rate)
            momentum_loss = momentum_conservation_loss(acc_pred, dt, momentum_loss_weight)
            
            combined_loss = (acc_loss_weight * acc_loss + 
                           temp_rate_loss_weight * temp_rate_loss + 
                           momentum_loss)
            
            total_loss += combined_loss.item()
            acc_loss_total += acc_loss.item()
            temp_rate_loss_total += temp_rate_loss.item()
            momentum_loss_total += momentum_loss.item()
            count += 1
    
    # Calculate averages
    avg_val_loss = total_loss / count if count > 0 else float('inf')
    avg_acc_loss = acc_loss_total / count if count > 0 else float('inf')
    avg_temp_rate_loss = temp_rate_loss_total / count if count > 0 else float('inf')
    avg_momentum_loss = momentum_loss_total / count if count > 0 else float('inf')
    
    # Return both the overall loss and component losses
    component_losses = {
        'acceleration': avg_acc_loss,
        'temp_rate': avg_temp_rate_loss,
        'momentum': avg_momentum_loss
    }
    
    return avg_val_loss, component_losses