import torch
import yaml
import argparse
from models.ctm import ContinuousThoughtMachine

class EnergyInference:
    def __init__(self, model_path, config_path, device='cpu'):
        # Load Config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        
        # Load Model
        # Reconstruct model args from config
        # Note: This assumes config structure matches __init__ args or we map them
        # For simplicity, we'll assume a flat config or specific mapping
        
        # Extract model params from config
        model_config = self.config
        
        self.model = ContinuousThoughtMachine(
            iterations=model_config['iterations'],
            d_model=model_config['d_model'],
            d_input=model_config['d_input'],
            heads=model_config['heads'],
            n_synch_out=model_config['n_synch_out'],
            n_synch_action=model_config['n_synch_action'],
            synapse_depth=model_config['synapse_depth'],
            memory_length=model_config['memory_length'],
            deep_nlms=model_config['deep_memory'],
            memory_hidden_dims=model_config['memory_hidden_dims'],
            do_layernorm_nlm=model_config['do_normalisation'],
            backbone_type=model_config['backbone_type'],
            positional_embedding_type=model_config['positional_embedding_type'],
            out_dims=model_config['out_dims'],
            prediction_reshaper=model_config.get('prediction_reshaper', [-1]),
            dropout=model_config.get('dropout', 0.0),
            neuron_select_type=model_config.get('neuron_select_type', 'random-pairing'),
            n_random_pairing_self=model_config.get('n_random_pairing_self', 0),
            energy_head_enabled=model_config.get('energy_head', {}).get('enabled', False),
            energy_hidden_dim=model_config.get('energy_head', {}).get('d_hidden', 64)
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def run_adaptive(self, inputs, energy_threshold=1.0, delta_threshold=0.01):
        """
        Runs the CTM and halts when Energy < threshold OR Energy stabilizes.
        """
        inputs = inputs.to(self.device)
        batch_size = inputs.shape[0]
        
        # We need to run the model step-by-step. 
        # However, the current CTM implementation runs the full loop in forward().
        # To support adaptive halting without refactoring the whole model into a cell,
        # we can run the full forward pass and then post-process the energy history 
        # to determine when it WOULD have halted.
        # This is less efficient but easier to implement given the current codebase.
        
        with torch.no_grad():
            # Run full forward pass
            predictions, certainties, energies = self.model(inputs)
            # energies shape: [B, 1, T]
            energies = energies.squeeze(1) # [B, T]
            
            final_predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            final_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            for b in range(batch_size):
                halted = False
                for t in range(self.model.iterations):
                    energy = energies[b, t]
                    
                    # 1. Check Absolute Energy Threshold
                    is_low_energy = energy < energy_threshold
                    
                    # 2. Check Convergence
                    if t > 0:
                        prev_energy = energies[b, t-1]
                        energy_delta = torch.abs(energy - prev_energy)
                        is_converged = energy_delta < delta_threshold
                    else:
                        is_converged = False
                        
                    if is_low_energy or is_converged:
                        final_predictions[b] = predictions[b, :, t].argmax()
                        final_steps[b] = t + 1
                        halted = True
                        break
                
                if not halted:
                    final_predictions[b] = predictions[b, :, -1].argmax()
                    final_steps[b] = self.model.iterations
                    
        return final_predictions, final_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    
    # Example usage (requires data)
    print("Inference script created. Use EnergyInference class to run adaptive inference.")
