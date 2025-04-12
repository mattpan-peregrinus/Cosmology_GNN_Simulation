import torch
import torch_geometric as pyg
from data_utils import preprocess

def validate(model, val_loader, device, loss_fn, acc_loss_weight, temp_loss_weight, metadata, noise_std, num_neighbors):
    model.eval()  
    total_loss = 0.0
    acc_loss_total = 0.0
    temp_loss_total = 0.0
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
                temperature_seq = batch["input"]["InternalEnergy"][i]
                
                graph = preprocess(
                    particle_type=None,
                    position_seq=input_coords,
                    target_position=target_coords,
                    metadata=metadata,
                    noise_std=noise_std,
                    num_neighbors=num_neighbors,
                    temperature_seq=temperature_seq,
                    target_temperature=batch["target"]["InternalEnergy"][i]
                )
                graphs.append(graph)
            
            if not graphs:
                continue
                
            batch_graph = pyg.data.Batch.from_data_list(graphs).to(device)
            
            # Forward pass
            predictions = model(batch_graph)
            acc_pred = predictions['acceleration']
            temp_pred = predictions['temperature']
            
            # Check if the attributes exist before computing loss
            if hasattr(batch_graph, 'y_acc') and batch_graph.y_acc is not None:
                acc_loss = loss_fn(acc_pred, batch_graph.y_acc)
            else:
                print("Warning: y_acc not found in validation graph")
                acc_loss = torch.tensor(0.0, device=device)
                
            if hasattr(batch_graph, 'y_temp') and batch_graph.y_temp is not None:
                temp_loss = loss_fn(temp_pred, batch_graph.y_temp)
            else:
                print("Warning: y_temp not found in validation graph")
                temp_loss = torch.tensor(0.0, device=device)
            
            combined_loss = acc_loss_weight * acc_loss + temp_loss_weight * temp_loss
            
            total_loss += combined_loss.item()
            acc_loss_total += acc_loss.item()
            temp_loss_total += temp_loss.item()
            count += 1
    
    # Calculate averages
    avg_val_loss = total_loss / count if count > 0 else float('inf')
    avg_acc_loss = acc_loss_total / count if count > 0 else float('inf')
    avg_temp_loss = temp_loss_total / count if count > 0 else float('inf')
    
    # Return both the overall loss and component losses
    component_losses = {
        'acceleration': avg_acc_loss,
        'temperature': avg_temp_loss
    }
    
    return avg_val_loss, component_losses