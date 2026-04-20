import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

# ---------------------------------------------------------
# Part 1: The Prunable Layers (Linear & Convolutional)
# ---------------------------------------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, layer_name="Linear"):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_name = layer_name
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 1.0) 

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity_loss(self):
        return torch.sum(torch.sigmoid(self.gate_scores))

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().flatten()

class PrunableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, layer_name="Conv"):
        super(PrunableConv2d, self).__init__()
        self.layer_name = layer_name
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate_scores = nn.Parameter(torch.Tensor(out_channels))
        nn.init.constant_(self.gate_scores, 1.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores).view(1, -1, 1, 1)
        return self.conv(x) * gates

    def get_sparsity_loss(self):
        return torch.sum(torch.sigmoid(self.gate_scores))

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().flatten()

# ---------------------------------------------------------
# Part 2: The CNN Architecture
# ---------------------------------------------------------
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()
        # Spatial Feature Extractors
        self.conv1 = PrunableConv2d(3, 16, layer_name="Conv1 (Edges)")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = PrunableConv2d(16, 32, layer_name="Conv2 (Textures)")
        
        # Dense Decision Makers
        self.fc1 = PrunableLinear(32 * 8 * 8, 256, layer_name="FC1 (Dense)")
        self.fc2 = PrunableLinear(256, 10, layer_name="FC2 (Output)")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_total_sparsity_loss(self):
        return sum(module.get_sparsity_loss() for module in self.modules() if isinstance(module, (PrunableLinear, PrunableConv2d)))

    def get_layerwise_gates(self):
        layer_gates = {}
        for module in self.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                layer_gates[module.layer_name] = module.get_gate_values()
        return layer_gates

# ---------------------------------------------------------
# Part 3: Training Loop
# ---------------------------------------------------------
# UPDATED: Added save_path parameter to save all 3 models!
def train_and_evaluate(target_lmbda, device, trainloader, testloader, epochs=15, save_path="pruned_model.pth"): 
    print(f"\n--- Training CNN with Target Lambda (λ) = {target_lmbda} ---")
    model = PrunableNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        current_lmbda = target_lmbda * min(1.0, epoch / (epochs * 0.7))

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            class_loss = criterion(outputs, labels)
            sparse_loss = model.get_total_sparsity_loss()
            total_loss = class_loss + current_lmbda * sparse_loss

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{epochs} | λ: {current_lmbda:.2e} | Loss: {running_loss/len(trainloader):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    layer_gates = model.get_layerwise_gates()
    threshold = 1e-2
    total_pruned, total_weights = 0, 0
    
    print("\n--- Layer-wise Sparsity Report ---")
    for name, gates in layer_gates.items():
        pruned = np.sum(gates < threshold)
        layer_total = len(gates)
        total_pruned += pruned
        total_weights += layer_total
        print(f"{name}:\t{100 * pruned / layer_total:.2f}% pruned")

    global_sparsity = 100 * total_pruned / total_weights
    print(f"\nFinal Test Accuracy: {accuracy:.2f}% | Global Sparsity: {global_sparsity:.2f}%")
    
    # UPDATED: Saves the model with the dynamic name we pass into the function
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained CNN model to '{save_path}'")

    return accuracy, global_sparsity, layer_gates

def plot_layerwise_distribution(layer_gates, lmbda):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Layer-wise Gate Distribution (λ = {lmbda})', fontsize=16)
    
    for ax, (name, gates) in zip(axes, layer_gates.items()):
        ax.hist(gates, bins=40, color='#2ca02c', edgecolor='black', alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Gate Value')
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    axes[0].set_ylabel('Frequency (Log Scale)')
    plt.tight_layout()
    plt.savefig('advanced_gate_distribution.png', dpi=300)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using compute engine: {device}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    lambdas = [0.0, 1e-3, 5e-3] 
    best_layer_gates = None
    final_summary = [] 

    for lmbda in lambdas:
        # NEW: Assign a specific file name based on the lambda value
        if lmbda == 0.0:
            save_name = "model_baseline.pth"
        elif lmbda == 1e-3:
            save_name = "model_medium.pth"
        else:
            save_name = "model_aggressive.pth"

        acc, spars, l_gates = train_and_evaluate(lmbda, device, trainloader, testloader, epochs=15, save_path=save_name)
        final_summary.append((lmbda, acc, spars)) 
        
        if lmbda > 0:
            best_layer_gates = l_gates

    if best_layer_gates:
        plot_layerwise_distribution(best_layer_gates, lambdas[-1])
        
    print("\n" + "="*50)
    print("🚀 FINAL 3-CYCLE SUMMARY REPORT FOR README")
    print("="*50)
    print(f"{'Lambda (λ)':<15} | {'Test Accuracy':<15} | {'Global Sparsity'}")
    print("-" * 50)
    for res in final_summary:
        print(f"{res[0]:<15.4f} | {res[1]:<13.2f} % | {res[2]:.2f} %")
    print("="*50)

if __name__ == '__main__':
    main()