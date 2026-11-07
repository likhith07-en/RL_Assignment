"""
Project 16: RL-based Learning-Rate Scheduler
Algorithm: Deep Q-Network (DQN) with Experience Replay & Target Network
Environment: CIFAR-10 Child CNN Training Loop
Baselines: Fixed LR, Heuristic StepLR, Random Action
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# -------------------------------------------------------------
# 1. Child Neural Network & Data Setup
# -------------------------------------------------------------
class ChildCNN(nn.Module):
    """Compact 3-layer CNN for fast CIFAR-10 training & evaluation."""
    def __init__(self, num_classes=10):
        super(ChildCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_cifar_loaders(data_dir="./data", batch_size=128, subset_size=5000):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    if subset_size is not None and subset_size < len(train_set):
        indices = list(range(subset_size))
        train_set = Subset(train_set, indices)
        test_indices = list(range(min(2000, len(test_set))))
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# -------------------------------------------------------------
# 2. RL Training Environment: CIFAR-10 LR Scheduler
# -------------------------------------------------------------
class LRSchedulerEnv:
    """
    Gym-like Environment wrapping the CNN training loop.
    State: [loss, delta_loss, val_acc, delta_val_acc, progress_ratio, log10_lr]
    Actions: 5 discrete LR multipliers [0.5, 0.8, 1.0, 1.25, 2.0]
    """
    ACTION_MULTIPLIERS = [0.5, 0.8, 1.0, 1.25, 2.0]

    def __init__(self, train_loader, test_loader, device, total_epochs=20, lr_init=0.01):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.total_epochs = total_epochs
        self.lr_init = lr_init
        self.lr_min = 1e-5
        self.lr_max = 0.2
        self.criterion = nn.CrossEntropyLoss()
        self.reset()

    def reset(self):
        self.model = ChildCNN().to(self.device)
        self.current_lr = self.lr_init
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.current_lr, momentum=0.9, weight_decay=5e-4)
        self.current_epoch = 0
        self.prev_loss = 2.5
        self.prev_val_acc = 0.10

        self.loss_history = []
        self.acc_history = []
        self.lr_history = [self.current_lr]

        state = np.array([
            2.5,          # Initial train loss
            0.0,          # Delta loss
            0.10,         # Val acc
            0.0,          # Delta val acc
            0.0,          # Progress ratio (epoch / total)
            np.log10(self.current_lr)
        ], dtype=np.float32)
        return state

    def step(self, action_idx):
        mult = self.ACTION_MULTIPLIERS[action_idx]
        self.current_lr = float(np.clip(self.current_lr * mult, self.lr_min, self.lr_max))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        # Train 1 epoch
        self.model.train()
        running_loss = 0.0
        total_batches = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / max(1, total_batches)

        # Validate
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = correct / max(1, total)

        self.current_epoch += 1
        self.loss_history.append(epoch_loss)
        self.acc_history.append(val_acc)
        self.lr_history.append(self.current_lr)

        # Reward Calculation
        delta_acc = val_acc - self.prev_val_acc
        delta_loss = epoch_loss - self.prev_loss
        reward = 100.0 * delta_acc - 5.0 * max(0.0, delta_loss)

        done = self.current_epoch >= self.total_epochs
        if done and val_acc >= 0.65:
            reward += 10.0 * (val_acc - 0.65)

        # Check divergence
        if np.isnan(epoch_loss) or epoch_loss > 15.0:
            reward -= 20.0
            done = True

        next_state = np.array([
            epoch_loss,
            delta_loss,
            val_acc,
            delta_acc,
            self.current_epoch / float(self.total_epochs),
            np.log10(self.current_lr)
        ], dtype=np.float32)

        self.prev_loss = epoch_loss
        self.prev_val_acc = val_acc

        info = {
            "epoch": self.current_epoch,
            "train_loss": epoch_loss,
            "val_acc": val_acc,
            "lr": self.current_lr
        }
        return next_state, reward, done, info

# -------------------------------------------------------------
# 3. DQN Agent & Experience Replay Buffer
# -------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=5):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim=6, action_dim=5, device="cpu", lr=1e-3, gamma=0.95):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.90

    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def update(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        curr_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# -------------------------------------------------------------
# 4. Evaluation Function for Baselines & Trained Agent
# -------------------------------------------------------------
def run_baseline_evaluation(env, baseline_type="fixed", agent=None):
    state = env.reset()
    done = False
    ep_rewards = []
    actions_taken = []

    step = 0
    while not done:
        if baseline_type == "fixed":
            # Action 2 is x1.0 (constant learning rate)
            action = 2
        elif baseline_type == "steplr":
            # Decay by x0.5 every 5 epochs, else keep x1.0
            action = 0 if (step + 1) % 5 == 0 else 2
        elif baseline_type == "random":
            action = random.randint(0, 4)
        elif baseline_type == "rl":
            action = agent.select_action(state, explore=False)
        else:
            action = 2

        next_state, reward, done, info = env.step(action)
        ep_rewards.append(reward)
        actions_taken.append(action)
        state = next_state
        step += 1

    return {
        "val_acc": env.acc_history,
        "train_loss": env.loss_history,
        "lr_history": env.lr_history,
        "total_reward": float(np.sum(ep_rewards)),
        "final_acc": float(env.acc_history[-1]) if env.acc_history else 0.0
    }

# -------------------------------------------------------------
# 5. Main Training & Benchmarking Pipeline
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RL-based Learning-Rate Scheduler (Project 16)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per training run")
    parser.add_argument("--rl_episodes", type=int, default=10, help="Number of RL training episodes")
    parser.add_argument("--subset", type=int, default=6000, help="CIFAR-10 training subset size for rapid training")
    parser.add_argument("--out_dir", type=str, default="16", help="Output directory for plots and models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Project 16] Device: {device} | SubsetSize: {args.subset} | Epochs: {args.epochs}")

    train_loader, test_loader = get_cifar_loaders(batch_size=128, subset_size=args.subset)
    env = LRSchedulerEnv(train_loader, test_loader, device=device, total_epochs=args.epochs, lr_init=0.01)

    agent = DQNAgent(state_dim=6, action_dim=5, device=device, lr=1e-3)

    # 1. Train DQN Agent
    print("\n>>> Phase 1: Training DQN Learning-Rate Controller...")
    dqn_rewards = []
    for ep in range(1, args.rl_episodes + 1):
        state = env.reset()
        done = False
        total_ep_reward = 0.0
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update(batch_size=32)
            state = next_state
            total_ep_reward += reward

        agent.decay_epsilon()
        if ep % 2 == 0:
            agent.sync_target()

        dqn_rewards.append(total_ep_reward)
        print(f"  RL Episode {ep:02d}/{args.rl_episodes:02d} | Reward: {total_ep_reward:6.2f} | Final ValAcc: {info['val_acc']*100:5.2f}% | Epsilon: {agent.epsilon:0.2f}")

    # Save Agent
    torch.save(agent.q_net.state_dict(), os.path.join(args.out_dir, "dqn_scheduler.pt"))
    print(f"Saved trained DQN model to {args.out_dir}/dqn_scheduler.pt")

    # 2. Benchmark Evaluation against 3 Baselines
    print("\n>>> Phase 2: Running Comparative Benchmark across 4 Policies...")
    results = {}

    print("  -> Testing Baseline 1: Fixed Learning Rate (eta = 0.01)...")
    results["Fixed LR"] = run_baseline_evaluation(env, baseline_type="fixed")

    print("  -> Testing Baseline 2: Heuristic Step Decay (StepLR x0.5 / 5 ep)...")
    results["StepLR Decay"] = run_baseline_evaluation(env, baseline_type="steplr")

    print("  -> Testing Baseline 3: Random Policy Scheduler...")
    results["Random Policy"] = run_baseline_evaluation(env, baseline_type="random")

    print("  -> Testing Evaluated Agent: Trained RL (DQN) Scheduler...")
    results["RL (DQN) Scheduler"] = run_baseline_evaluation(env, baseline_type="rl", agent=agent)

    # 3. Print Summary Table
    print("\n" + "="*65)
    print(f"{'Method':<25} | {'Final Val Acc (%)':<18} | {'Total Reward':<12}")
    print("="*65)
    for name, data in results.items():
        print(f"{name:<25} | {data['final_acc']*100:<18.2f} | {data['total_reward']:<12.2f}")
    print("="*65)

    # Save Results JSON
    summary_path = os.path.join(args.out_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: {"final_acc": v["final_acc"], "total_reward": v["total_reward"], "lr_history": v["lr_history"]} for k, v in results.items()}, f, indent=2)

    # 4. Generate High-Quality Comparative Learning Curves Plot
    plot_path = os.path.join(args.out_dir, "learning_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    colors = {"Fixed LR": "#7f7f7f", "StepLR Decay": "#1f77b4", "Random Policy": "#e377c2", "RL (DQN) Scheduler": "#2ca02c"}
    styles = {"Fixed LR": "--", "StepLR Decay": "-.", "Random Policy": ":", "RL (DQN) Scheduler": "-"}

    # Left: Validation Accuracy vs Epochs
    epochs = list(range(1, args.epochs + 1))
    for name, data in results.items():
        ax1.plot(epochs, [a * 100 for a in data["val_acc"]], label=name, color=colors[name], linestyle=styles[name], linewidth=2.2)

    ax1.set_title("Validation Accuracy vs. Epochs on CIFAR-10", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=10)

    # Right: Learning Rate Trajectories over Time
    for name, data in results.items():
        # lr_history has length total_epochs + 1
        ax2.plot(range(len(data["lr_history"])), data["lr_history"], label=name, color=colors[name], linestyle=styles[name], linewidth=2.0)

    ax2.set_yscale("log")
    ax2.set_title("Dynamic Learning-Rate Schedules (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Step (Epoch)", fontsize=11)
    ax2.set_ylabel("Learning Rate (η)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[Project 16] Successfully generated learning curves plot at {plot_path}")

if __name__ == "__main__":
    main()

