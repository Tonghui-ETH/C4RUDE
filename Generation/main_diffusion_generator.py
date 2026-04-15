import os
import math
import argparse
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Repository root is resolved automatically from this script's location.
# Structure: <repo_root>/Code for C4RUDE/Generation/<script>.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Configuration ---
@dataclass
class TrainingConfig:
    image_size = 128  # The size of the input images
    train_batch_size = 8  # Increased batch size for better GPU utilization
    eval_batch_size = 8   # Increased batch size
    num_epochs = 200
    learning_rate = 1e-4  # Standard diffusion model learning rate
    lr_warmup_steps = 500
    save_image_epochs = 5  # More frequent to monitor quality
    save_model_epochs = 10  # More frequent checkpoints
    num_inference_steps = 200  # More steps for better quality
    num_dataloader_workers = 8  # Increased workers for faster data loading
    output_dir = os.path.join(REPO_ROOT, "results", "Generation")
    gpu_id = 0  # Use GPU 1 (change to 2 or 3 if needed)
    seed = 42
    use_grayscale = False
    stats_sample_limit = None  # Optional cap when computing dataset statistics
    channel_mean = None
    channel_std = None
    
    # Classifier-Free Guidance parameters
    conditioning_dropout_prob = 0.1  # 10% chance to drop conditioning during training
    guidance_scale = 7.5  # Strength of guidance during inference (higher = more conditioning influence)
    embedding_dim = 256  # Increased from 128 for better concentration discrimination

    # Data paths
    train_root = os.path.join(REPO_ROOT, "Data", "01 needle", "prediction", "concentration_at_frequency_35kHz", "train")
    val_root   = os.path.join(REPO_ROOT, "Data", "01 needle", "prediction", "concentration_at_frequency_35kHz", "val")
    
    # Concentrations
    # --- USER CONFIGURATION: SPLIT SETTINGS ---
    # 1. Define ALL concentrations available in your dataset
    all_available_concentrations = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    # 2. Select the concentration to HOLD OUT for generation (testing)
    # The rest will be used for training.
    test_concentration = 3.0  # Changed from 8.0 to hold out concentration 5
    
    # (Do not modify these manually, they are calculated automatically)
    known_concentrations = None
    unknown_concentrations_for_generation = None

    def __post_init__(self):
        # Automatic splitting logic
        if self.known_concentrations is None:
            self.known_concentrations = [c for c in self.all_available_concentrations if c != self.test_concentration]
        
        if self.unknown_concentrations_for_generation is None:
            self.unknown_concentrations_for_generation = [self.test_concentration]
            
        # Update output directory to create a unique folder for this test concentration
        # This prevents overwriting models when you switch test concentrations.
        self.output_dir = os.path.join(self.output_dir, f"holdout_{str(self.test_concentration).replace('.', '_')}")
        
        print(f"--- Configuration Setup ---")
        print(f"Training on concentrations: {self.known_concentrations}")
        print(f"Testing (Generating) on: {self.unknown_concentrations_for_generation}")
        print(f"Output Directory: {self.output_dir}")
        print(f"---------------------------")

        values = torch.tensor(self.known_concentrations, dtype=torch.float32)
        self.concentration_mean = values.mean().item()
        std = values.std(unbiased=False).item()
        self.concentration_std = std if std > 0 else 1.0

config = TrainingConfig()


# --- Custom Dataset ---
# We can reuse parts of your existing data loading logic

class CenterCircleMask:
    def __init__(self, radius_ratio: float = 0.25):
        self.radius_ratio = radius_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        r = int(min(w, h) * self.radius_ratio)
        center = (w // 2, h // 2)
        draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), fill=255)
        img = img.copy()
        img.putalpha(mask)
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img)
        return img.convert("RGB")

def collect_concentration_samples(root_dir, allowed_concentrations=None):
    image_paths = []
    labels = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    for conc_folder in os.listdir(root_dir):
        try:
            concentration = float(conc_folder)
            if allowed_concentrations and concentration not in allowed_concentrations:
                continue
            folder_path = os.path.join(root_dir, conc_folder)
            if os.path.isdir(folder_path):
                for fname in os.listdir(folder_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(folder_path, fname))
                        labels.append(concentration)
        except ValueError:
            print(f"Skipping non-numeric folder: {conc_folder}")

    return image_paths, labels


class ConcentrationDataset(Dataset):
    def __init__(self, root_dir, transform=None, allowed_concentrations=None):
        self.transform = transform
        self.image_paths, self.labels = collect_concentration_samples(root_dir, allowed_concentrations)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
            
        return image, label


def summarize_dataset(dataset: ConcentrationDataset, name: str) -> None:
    """Print how many samples exist for each concentration."""
    if dataset is None or len(dataset) == 0:
        print(f"{name}: no samples found.")
        return

    counts = Counter(dataset.labels)
    total = len(dataset)
    print(f"{name}: {total} samples")
    for conc in sorted(counts.keys()):
        share = counts[conc] / total * 100
        print(f"  concentration {conc:g}: {counts[conc]} ({share:.2f}%)")


def compute_channel_stats(image_paths, image_size, use_grayscale=False, sample_limit=None):
    if not image_paths:
        raise ValueError("No images available to compute statistics.")

    num_channels = 1 if use_grayscale else 3
    accumulator = torch.zeros(num_channels, dtype=torch.float64)
    accumulator_sq = torch.zeros(num_channels, dtype=torch.float64)
    total_pixels = 0

    resize_to_tensor = [transforms.Resize((image_size, image_size))]
    if use_grayscale:
        resize_to_tensor.append(transforms.Grayscale(num_output_channels=1))
    resize_to_tensor.append(transforms.ToTensor())
    prep = transforms.Compose(resize_to_tensor)

    for idx, img_path in enumerate(image_paths):
        if sample_limit is not None and idx >= sample_limit:
            break
        tensor = prep(Image.open(img_path).convert("RGB"))
        tensor = tensor.view(num_channels, -1)
        accumulator += tensor.sum(dim=1)
        accumulator_sq += (tensor ** 2).sum(dim=1)
        total_pixels += tensor.shape[1]

    mean = accumulator / total_pixels
    var = accumulator_sq / total_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.tolist(), std.tolist()


def normalize_concentrations(values: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """Scale concentration values for numerically stable conditioning."""
    return (values - config.concentration_mean) / config.concentration_std


def plot_training_metrics(train_losses, val_losses, learning_rates, epochs, save_path):
    """Plot and save training metrics including loss curves and learning rate."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    if val_losses:
        valid_val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        valid_val_losses = [v for v in val_losses if v is not None]
        ax1.plot(valid_val_epochs, valid_val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visibility
    
    # Plot 2: Learning Rate Schedule
    ax2 = axes[1]
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for learning rate
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training metrics plot saved to {save_path}")


def plot_loss_comparison(train_losses, val_losses, epochs, save_path):
    """Create a detailed comparison plot of training vs validation loss."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot training loss
    ax.plot(epochs, train_losses, 'b-o', label='Training Loss', 
            linewidth=2, markersize=4, alpha=0.7)
    
    # Plot validation loss
    if val_losses:
        valid_val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
        valid_val_losses = [v for v in val_losses if v is not None]
        ax.plot(valid_val_epochs, valid_val_losses, 'r-s', label='Validation Loss',
                linewidth=2, markersize=4, alpha=0.7)
        
        # Mark best validation loss
        if valid_val_losses:
            best_val_loss = min(valid_val_losses)
            best_epoch = valid_val_epochs[valid_val_losses.index(best_val_loss)]
            ax.plot(best_epoch, best_val_loss, 'r*', markersize=20, 
                   label=f'Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=13, fontweight='bold')
    ax.set_title('Training Progress: Loss Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Loss comparison plot saved to {save_path}")


# --- Comparison Stitching Function ---
@torch.no_grad()
def generate_comparison_stitched_images(model, cond_embedding, noise_scheduler, dataset, config, device, epoch):
    """Generate side-by-side comparison of real vs generated images for one set from each concentration."""
    print(f"\n--- Creating comparison stitched images at epoch {epoch} ---")
    model.eval()
    cond_embedding.eval()
    
    comparison_dir = os.path.join(config.output_dir, "comparison_stitched")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Get one real image from each training concentration
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    real_images_dict = {c: None for c in config.known_concentrations}
    
    print("Collecting one real image per concentration...")
    for imgs, lbls in loader:
        c = float(lbls[0].item())
        if c in real_images_dict and real_images_dict[c] is None:
            img = imgs[0].to(device)
            # Denormalize to [0,1]
            if config.channel_mean is not None:
                mean = torch.tensor(config.channel_mean, device=device).view(-1, 1, 1)
                std = torch.tensor(config.channel_std, device=device).view(-1, 1, 1)
                img = img * std + mean
            real_images_dict[c] = img.clamp(0, 1)
        
        # Break if we have one sample for each concentration
        if all(v is not None for v in real_images_dict.values()):
            break
    
    # Generate one image for each concentration
    all_real = []
    all_generated = []
    conc_labels = []
    
    for conc in sorted(config.known_concentrations):
        if real_images_dict[conc] is None:
            print(f"Warning: No real image found for concentration {conc}")
            continue
        
        print(f"Generating image for concentration {conc}...")
        
        # Add real image
        all_real.append(real_images_dict[conc])
        conc_labels.append(conc)
        
        # Generate one synthetic image with CFG
        noise_scheduler.set_timesteps(config.num_inference_steps)
        generator = torch.Generator(device=device).manual_seed(config.seed + int(conc * 100) + epoch)
        
        syn_image = torch.randn(
            (1, model.config.in_channels, model.config.sample_size, model.config.sample_size),
            device=device, generator=generator
        )
        
        # Prepare conditioning
        cond_input = normalize_concentrations(torch.tensor([conc], device=device), config)
        cond_emb = cond_embedding(cond_input).unsqueeze(1)
        # null_embedding is [1, 256], unsqueeze(1) -> [1, 1, 256]
        uncond_emb = cond_embedding.null_embedding.unsqueeze(1)
        
        # Denoising with CFG
        for t in noise_scheduler.timesteps:
            syn_input = torch.cat([syn_image, syn_image], dim=0)
            emb_input = torch.cat([cond_emb, uncond_emb], dim=0)
            noise_pred = model(syn_input, t, encoder_hidden_states=emb_input).sample
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            syn_image = noise_scheduler.step(noise_pred, t, syn_image).prev_sample
        
        # Denormalize
        if config.channel_mean is not None:
            mean = torch.tensor(config.channel_mean, device=device).view(1, -1, 1, 1)
            std = torch.tensor(config.channel_std, device=device).view(1, -1, 1, 1)
            syn_image = syn_image * std + mean
        syn_image = syn_image.clamp(0, 1)
        
        all_generated.append(syn_image[0])
    
    # Create stitched comparison image
    if all_real and all_generated:
        # Stack: Top row = Real images, Bottom row = Generated images
        real_row = torch.stack(all_real)
        gen_row = torch.stack(all_generated)
        stitched = torch.cat([real_row, gen_row], dim=0)
        
        # Save stitched image
        save_path = os.path.join(comparison_dir, f"comparison_epoch_{epoch:04d}.png")
        save_image(stitched, save_path, nrow=len(all_real), normalize=False, padding=2, pad_value=1.0)
        
        print(f"✓ Saved comparison stitched image to {save_path}")
        print(f"  Layout: Top row = Real, Bottom row = Generated")
        print(f"  Concentrations (left to right): {conc_labels}")
    
    model.train()
    cond_embedding.train()


# --- Generation Function ---
@torch.no_grad()
def generate_images(model, cond_embedding, noise_scheduler, concentrations, epoch, config, device):
    """Generates and saves a grid of images for the given concentrations using Classifier-Free Guidance."""
    model.eval()
    cond_embedding.eval()
    
    # Create embeddings for the concentrations we want to generate
    target_concentrations = torch.tensor(concentrations, dtype=torch.float32, device=device)
    cond_inputs = normalize_concentrations(target_concentrations, config)
    cond_embeds = cond_embedding(cond_inputs).unsqueeze(1)
    
    # Create unconditional embeddings (null conditioning)
    # null_embedding is [1, 256], expand to [n, 256], then unsqueeze(1) -> [n, 1, 256]
    uncond_embeds = cond_embedding.null_embedding.expand(len(concentrations), -1).unsqueeze(1)

    # Start with random noise
    generator = torch.Generator(device=device).manual_seed(config.seed)
    
    images = torch.randn(
        (len(concentrations), model.config.in_channels, model.config.sample_size, model.config.sample_size),
        device=device,
        generator=generator,
    )

    # Set the number of inference steps
    noise_scheduler.set_timesteps(config.num_inference_steps)

    # Denoising loop with Classifier-Free Guidance
    for t in tqdm(noise_scheduler.timesteps, desc=f"Generating (CFG={config.guidance_scale})"):
        # Duplicate images for conditional and unconditional prediction
        images_input = torch.cat([images, images], dim=0)
        t_input = t
        
        # Conditional and unconditional embeddings
        encoder_hidden_states = torch.cat([cond_embeds, uncond_embeds], dim=0)
        
        # Predict noise
        noise_pred = model(images_input, t_input, encoder_hidden_states=encoder_hidden_states).sample
        
        # Split predictions
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        
        # Apply Classifier-Free Guidance
        # noise_pred = uncond + guidance_scale * (cond - uncond)
        noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Compute previous image: x_t -> x_t-1
        images = noise_scheduler.step(noise_pred, t, images).prev_sample

    # Denormalize and save
    # We must reverse the normalization applied during training: x = z * std + mean
    if config.channel_mean is not None and config.channel_std is not None:
        mean = torch.tensor(config.channel_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(config.channel_std, device=device).view(1, -1, 1, 1)
        images = (images * std + mean)
    else:
        # Fallback if stats are missing (should not happen if main() is run)
        # Assumes images were normalized to [-1, 1]
        images = images / 2 + 0.5

    images = images.clamp(0, 1)
    
    base_samples_dir = os.path.join(config.output_dir, "samples")
    for idx, (conc, img_tensor) in enumerate(zip(concentrations, images), start=1):
        # Create a separate folder for each concentration (e.g. "3.0")
        conc_folder = str(conc)
        conc_dir = os.path.join(base_samples_dir, conc_folder)
        os.makedirs(conc_dir, exist_ok=True)

        conc_str = str(conc).replace(".", "_")
        filename = f"epoch_{epoch:04d}_conc_{conc_str}_{idx}.png"
        save_path = os.path.join(conc_dir, filename)
        save_image(img_tensor, save_path, normalize=False)
    
    print(f"Generated images saved to {base_samples_dir} organized by concentration folders")

    model.train()
    cond_embedding.train()


# --- Verification & Quality Check Functions ---
@torch.no_grad()
def verify_model_quality(model, cond_embedding, noise_scheduler, dataset, config, device):
    """
    Performs two checks to verify model quality:
    1. Side-by-side comparison of Real vs Generated images (Visual Statistical Check).
    2. Reconstruction test: Adds noise to real images and asks model to fix them (Structure Understanding Check).
    """
    print("\n--- Running Model Verification ---")
    model.eval()
    cond_embedding.eval()
    save_dir = os.path.join(config.output_dir, "verification")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Real vs Generated Comparison
    # Modified: Use ALL known (training) concentrations instead of just first 3
    test_concs = config.known_concentrations
    
    print(f"Generating comparison images for all training concentrations: {test_concs}")
    
    loader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=True)
    real_images_dict = {c: [] for c in test_concs}
    
    # Collect some real images for each training concentration
    print("Collecting real images from training set...")
    for imgs, lbls in loader:
        for i in range(len(lbls)):
            c = float(lbls[i].item())
            if c in real_images_dict and len(real_images_dict[c]) < 4:
                # Denormalize map to [0,1]
                img = imgs[i].to(device)
                if config.channel_mean is not None:
                     mean = torch.tensor(config.channel_mean, device=device).view(-1, 1, 1)
                     std = torch.tensor(config.channel_std, device=device).view(-1, 1, 1)
                     img = img * std + mean
                real_images_dict[c].append(img.clamp(0, 1))
        if all(len(v) >= 4 for v in real_images_dict.values()):
            break

    # Generate matching synthetic images for ALL training concentrations
    for conc in test_concs:
        if len(real_images_dict[conc]) == 0:
            print(f"Warning: No real images found for concentration {conc}, skipping...")
            continue
        
        print(f"Generating comparison for concentration {conc}...")
        
        # Prepare real row
        real_row = torch.stack(real_images_dict[conc])
        
        # Prepare generated row (same number as real) with CFG
        noise_scheduler.set_timesteps(config.num_inference_steps)
        generator = torch.Generator(device=device).manual_seed(config.seed + int(conc))
        
        syn_images = torch.randn(
            (len(real_row), model.config.in_channels, model.config.sample_size, model.config.sample_size),
            device=device, generator=generator
        )
        cond_input = normalize_concentrations(torch.full((len(real_row),), conc, device=device), config)
        cond_emb = cond_embedding(cond_input).unsqueeze(1)
        # null_embedding is [1, 256], expand to [n, 256], then unsqueeze(1) -> [n, 1, 256]
        uncond_emb = cond_embedding.null_embedding.expand(len(real_row), -1).unsqueeze(1)
        
        for t in tqdm(noise_scheduler.timesteps, desc=f"Gen conc {conc}", leave=False):
            # CFG: predict with both conditional and unconditional
            syn_input = torch.cat([syn_images, syn_images], dim=0)
            emb_input = torch.cat([cond_emb, uncond_emb], dim=0)
            noise_pred = model(syn_input, t, encoder_hidden_states=emb_input).sample
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            syn_images = noise_scheduler.step(noise_pred, t, syn_images).prev_sample
            
        if config.channel_mean is not None:
             mean = torch.tensor(config.channel_mean, device=device).view(1, -1, 1, 1)
             std = torch.tensor(config.channel_std, device=device).view(1, -1, 1, 1)
             syn_images = syn_images * std + mean
        syn_images = syn_images.clamp(0, 1)

        # Stitch: Top Row = Real, Bottom Row = Generated
        comparison = torch.cat([real_row, syn_images], dim=0)
        save_path = os.path.join(save_dir, f"compare_conc_{str(conc).replace('.', '_')}.png")
        save_image(comparison, save_path, nrow=len(real_row), normalize=False)
        print(f"✓ Saved comparison for concentration {conc} to {save_path}")

    # Create a summary grid showing all concentrations side-by-side
    print("\nCreating master comparison grid...")
    all_comparisons = []
    for conc in sorted(test_concs):
        if len(real_images_dict[conc]) > 0:
            # Take first 2 images from each concentration for the summary
            real_sample = real_images_dict[conc][:2]
            
            # Generate 2 synthetic images with CFG
            noise_scheduler.set_timesteps(config.num_inference_steps)
            generator = torch.Generator(device=device).manual_seed(config.seed + int(conc * 10))
            
            syn_images = torch.randn(
                (2, model.config.in_channels, model.config.sample_size, model.config.sample_size),
                device=device, generator=generator
            )
            cond_input = normalize_concentrations(torch.full((2,), conc, device=device), config)
            cond_emb = cond_embedding(cond_input).unsqueeze(1)
            # null_embedding is [1, 256], expand to [2, 256], then unsqueeze(1) -> [2, 1, 256]
            uncond_emb = cond_embedding.null_embedding.expand(2, -1).unsqueeze(1)
            
            for t in noise_scheduler.timesteps:
                # CFG: predict with both conditional and unconditional
                syn_input = torch.cat([syn_images, syn_images], dim=0)
                emb_input = torch.cat([cond_emb, uncond_emb], dim=0)
                noise_pred = model(syn_input, t, encoder_hidden_states=emb_input).sample
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                syn_images = noise_scheduler.step(noise_pred, t, syn_images).prev_sample
                
            if config.channel_mean is not None:
                 mean = torch.tensor(config.channel_mean, device=device).view(1, -1, 1, 1)
                 std = torch.tensor(config.channel_std, device=device).view(1, -1, 1, 1)
                 syn_images = syn_images * std + mean
            syn_images = syn_images.clamp(0, 1)
            
            # Stack: Real1, Generated1, Real2, Generated2
            pair = torch.stack([real_sample[0], syn_images[0], real_sample[1], syn_images[1]])
            all_comparisons.append(pair)
    
    if all_comparisons:
        # Create grid: each row is one concentration, columns alternate Real/Gen
        master_grid = torch.cat(all_comparisons, dim=0)
        master_path = os.path.join(save_dir, "all_concentrations_summary.png")
        save_image(master_grid, master_path, nrow=4, normalize=False)
        print(f"✓ Saved master summary grid to {master_path}")
        print(f"  Grid layout: Each row = one concentration (Real1, Gen1, Real2, Gen2)")

    # 2. Reconstruction / Inpainting Test
    # Take real images, add partial noise (e.g. 50%), then denoise.
    # This checks if the model respects the underlying structure of training data.
    print("\nRunning Reconstruction Test (Denoising from 50% noise)...")
    try:
        data_iter = iter(loader)
        real_imgs, real_lbls = next(data_iter)
        real_imgs = real_imgs[:4].to(device)
        real_lbls = real_lbls[:4].to(device)
        
        # Add noise to t = max_steps / 2
        start_timestep = config.num_inference_steps // 2
        noise = torch.randn_like(real_imgs)
        # map inference steps to train timesteps for add_noise
        train_t = (start_timestep * noise_scheduler.config.num_train_timesteps) // config.num_inference_steps
        timesteps = torch.full((real_imgs.shape[0],), train_t, device=device, dtype=torch.long)
        
        noisy_imgs = noise_scheduler.add_noise(real_imgs, noise, timesteps)
        
        # Denoise starting from this noisy state
        curr_imgs = noisy_imgs.clone()
        
        # Prepare conditioning
        cond_input = normalize_concentrations(real_lbls, config)
        cond_emb = cond_embedding(cond_input).unsqueeze(1)
        
        # We need to find the correct start point in the scheduler's timesteps list
        # Assuming timesteps are descending [999, ..., 0]
        # We want to start loop from the step corresponding to 'start_timestep'
        metrics_timesteps = noise_scheduler.timesteps
        # Find index close to train_t
        # This is approximate mapping for DDPMScheduler or DDIMScheduler
        start_idx = len(metrics_timesteps) - start_timestep  # rough approximation if linear
        # Better: iterate and verify
        
        noise_scheduler.set_timesteps(config.num_inference_steps)
        # We only run the last 'start_timestep' steps
        relevant_timesteps = noise_scheduler.timesteps[-start_timestep:]
        
        for t in tqdm(relevant_timesteps, desc="Reconstructing", leave=False):
            noise_pred = model(curr_imgs, t, encoder_hidden_states=cond_emb).sample
            curr_imgs = noise_scheduler.step(noise_pred, t, curr_imgs).prev_sample
            
        # Denormalize for display
        if config.channel_mean is not None:
             mean = torch.tensor(config.channel_mean, device=device).view(1, -1, 1, 1)
             std = torch.tensor(config.channel_std, device=device).view(1, -1, 1, 1)
             
             disp_real = (real_imgs * std + mean).clamp(0, 1)
             disp_noisy = (noisy_imgs * std + mean).clamp(0, 1)
             disp_recon = (curr_imgs * std + mean).clamp(0, 1)
        
        # Save: Row 1 Real, Row 2 Noisy, Row 3 Reconstructed
        recon_grid = torch.cat([disp_real, disp_noisy, disp_recon], dim=0)
        save_path = os.path.join(save_dir, "reconstruction_test.png")
        save_image(recon_grid, save_path, nrow=4, normalize=False)
        print(f"✓ Saved reconstruction test to {save_path}")
        
    except Exception as e:
        print(f"⚠ Skipping reconstruction test due to error: {e}")

    print("\n--- Verification Complete ---")
    print(f"All verification results saved to: {save_dir}")
    model.train()


# --- Conditioning Embedding ---
# Enhanced embedding with Sinusoidal Positional Encoding for better scalar discrimination
class SinusoidalPositionalEmbedding(nn.Module):
    """Maps a scalar value to a high-dimensional vector using sinusoidal frequencies.
    This is critical for distinguishing similar values like 2.0 vs 3.0."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # x: (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        # Scale concentration to spread out in frequency space
        emb = x * 1000.0 * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, dim)
        return emb


class ConcentrationEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.sinusoidal = SinusoidalPositionalEmbedding(out_features)
        self.projection = nn.Sequential(
            nn.Linear(out_features + 1, out_features * 2),  # sinusoidal + raw value
            nn.LayerNorm(out_features * 2),
            nn.SiLU(),
            nn.Linear(out_features * 2, out_features * 2),
            nn.LayerNorm(out_features * 2),
            nn.SiLU(),
            nn.Linear(out_features * 2, out_features),
            nn.LayerNorm(out_features),
        )
        # Initialize null embedding with small random values (not zeros!)
        self.null_embedding = nn.Parameter(torch.randn(1, out_features) * 0.01)

    def forward(self, x, mask=None):
        # Reshape x from (batch_size,) to (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # Get sinusoidal features + raw value
        sin_features = self.sinusoidal(x)  # (batch_size, out_features)
        combined = torch.cat([sin_features, x], dim=-1)  # (batch_size, out_features + 1)
        cond_emb = self.projection(combined)
        
        # Apply mask for classifier-free guidance training
        if mask is not None:
            null_emb = self.null_embedding.expand(x.shape[0], -1)
            mask_expanded = mask.unsqueeze(1).float()
            cond_emb = cond_emb * (1 - mask_expanded) + null_emb * mask_expanded
        
        return cond_emb

# --- Training Function ---
def evaluate(model, cond_embedding, noise_scheduler, val_dataloader, config):
    """Run a single pass over the validation set to monitor diffusion loss."""
    if val_dataloader is None:
        return None

    model.eval()
    cond_embedding.eval()
    
    # Get device from model
    device = next(model.parameters()).device

    losses = []
    with torch.no_grad():
        for clean_images, concentrations in val_dataloader:
            # Move data to device
            clean_images = clean_images.to(device)
            concentrations = concentrations.to(device)
            
            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Apply conditioning dropout for Classifier-Free Guidance training
            cond_inputs = normalize_concentrations(concentrations, config)
            dropout_mask = torch.rand(bs, device=clean_images.device) < config.conditioning_dropout_prob
            cond_embeds = cond_embedding(cond_inputs, mask=dropout_mask).unsqueeze(1)
            
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states=cond_embeds).sample
            loss = F.mse_loss(noise_pred, noise)
            losses.append(loss.detach())

    model.train()
    cond_embedding.train()

    if not losses:
        return None

    loss_tensor = torch.stack(losses)
    return loss_tensor.mean().item()


def train_loop(
    config,
    model,
    cond_embedding,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    val_dataloader=None,
):
    # Setup device - use specified GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"🚀 Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {config.train_batch_size}")
    print(f"Number of CPU workers: {config.num_dataloader_workers}")
    print(f"{'='*60}\n")
    
    # Move models to device
    model = model.to(device)
    cond_embedding = cond_embedding.to(device)
    
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    
    # Training history tracking
    train_losses_per_epoch = []
    val_losses_per_epoch = []
    learning_rates_per_epoch = []
    epoch_numbers = []

    # Now you train the model
    for epoch in range(config.num_epochs):
        epoch_train_losses = []
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images, concentrations = batch
            clean_images = clean_images.to(device)
            concentrations = concentrations.to(device)
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Get the concentration embedding with conditioning dropout
            cond_inputs = normalize_concentrations(concentrations, config)
            dropout_mask = torch.rand(bs, device=device) < config.conditioning_dropout_prob
            cond_embeds = cond_embedding(cond_inputs, mask=dropout_mask) # shape: (batch_size, embed_dim)
            # The UNet expects the conditioning in shape (batch_size, sequence_length, embed_dim)
            cond_embeds = cond_embeds.unsqueeze(1) # shape: (batch_size, 1, embed_dim)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states=cond_embeds).sample

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            # Clip gradients for ALL parameters (model + embedding)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(cond_embedding.parameters()), 1.0
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Record training loss
            epoch_train_losses.append(loss.detach().item())

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # After each epoch, record metrics
        avg_train_loss = np.mean(epoch_train_losses)
        current_lr = lr_scheduler.get_last_lr()[0]
        
        epoch_numbers.append(epoch)
        train_losses_per_epoch.append(avg_train_loss)
        learning_rates_per_epoch.append(current_lr)
        
        # Evaluate and record validation loss
        val_loss = None
        if val_dataloader is not None:
            val_loss = evaluate(model, cond_embedding, noise_scheduler, val_dataloader, config)
        val_losses_per_epoch.append(val_loss)

        if val_loss is not None:
            print(f"Validation loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.6f}. Saving best model...")
                
                best_dir = os.path.join(config.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                
                model.save_pretrained(best_dir)
                torch.save(cond_embedding.state_dict(), os.path.join(best_dir, "cond_embedding.pth"))
                
                # Generate images with the best model to track quality improvement
                print(f"Generating images with best model (epoch {epoch+1})...")
                concentrations_to_generate = config.all_available_concentrations
                generate_images(
                    model, cond_embedding, noise_scheduler, concentrations_to_generate, 
                    epoch + 1, config, device
                )

        # Generate comparison stitched images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
            generate_comparison_stitched_images(
                model, cond_embedding, noise_scheduler, train_dataloader.dataset, config, device, epoch + 1
            )
        
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            # Generate images for all concentrations (both known and unknown)
            concentrations_to_generate = config.all_available_concentrations
            generate_images(
                model, cond_embedding, noise_scheduler, concentrations_to_generate, epoch + 1, config, device
            )
        
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # Save the UNet model
            model.save_pretrained(config.output_dir)
            
            # Save the conditioning embedding model
            torch.save(cond_embedding.state_dict(), os.path.join(config.output_dir, "cond_embedding.pth"))
            print(f"Model saved to {config.output_dir}")
            
            # Plot and save training metrics
            metrics_dir = os.path.join(config.output_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            plot_training_metrics(
                train_losses_per_epoch, val_losses_per_epoch, learning_rates_per_epoch,
                epoch_numbers, os.path.join(metrics_dir, f"training_metrics_epoch_{epoch+1:04d}.png")
            )
            
            plot_loss_comparison(
                train_losses_per_epoch, val_losses_per_epoch, epoch_numbers,
                os.path.join(metrics_dir, f"loss_comparison_epoch_{epoch+1:04d}.png")
            )
    
    # Final plots at end of training
    print("\n--- Saving Final Training Metrics ---")
    metrics_dir = os.path.join(config.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    plot_training_metrics(
        train_losses_per_epoch, val_losses_per_epoch, learning_rates_per_epoch,
        epoch_numbers, os.path.join(metrics_dir, "training_metrics_final.png")
    )
    
    plot_loss_comparison(
        train_losses_per_epoch, val_losses_per_epoch, epoch_numbers,
        os.path.join(metrics_dir, "loss_comparison_final.png")
    )


# --- Main Script ---
def main():
    # 1. DATA PREPARATION
    train_dataset = ConcentrationDataset(
        root_dir=config.train_root,
        transform=None,
        allowed_concentrations=config.known_concentrations,
    )

    if config.channel_mean is None or config.channel_std is None:
        mean, std = compute_channel_stats(
            train_dataset.image_paths,
            image_size=config.image_size,
            use_grayscale=config.use_grayscale,
            sample_limit=config.stats_sample_limit,
        )
        config.channel_mean = mean
        config.channel_std = std
        print(f"Computed dataset mean: {mean}")
        print(f"Computed dataset std: {std}")
    else:
        mean, std = config.channel_mean, config.channel_std

    preprocess_transforms = [
        transforms.Resize((config.image_size, config.image_size)),
    ]
    if config.use_grayscale:
        preprocess_transforms.append(transforms.Grayscale(num_output_channels=1))
    preprocess_transforms.append(transforms.ToTensor())
    preprocess_transforms.append(transforms.Normalize(mean, std))
    preprocess = transforms.Compose(preprocess_transforms)

    train_dataset.transform = preprocess
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.num_dataloader_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True  # Keep workers alive between epochs
    )

    val_dataloader = None
    if os.path.isdir(config.val_root):
        val_dataset = ConcentrationDataset(
            root_dir=config.val_root,
            transform=preprocess,
            allowed_concentrations=config.known_concentrations,
        )
        if len(val_dataset) > 0:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=config.num_dataloader_workers,
                pin_memory=True,
                persistent_workers=True
            )
        else:
            print("Validation dataset is empty. Skipping validation phase.")
    else:
        print(f"Validation directory not found at {config.val_root}. Skipping validation phase.")

    summarize_dataset(train_dataset, "Train dataset")
    if val_dataloader is not None:
        summarize_dataset(val_dataloader.dataset, "Validation dataset")
    print(f"Known concentrations for training: {config.known_concentrations}")
    print(f"Concentrations for generation: {config.unknown_concentrations_for_generation}")

    # 2. MODEL DEFINITION
    # The UNet model for denoising
    # CRITICAL: Use CrossAttention at MULTIPLE resolution levels
    # so the model can effectively "see" the concentration condition
    in_out_channels = 1 if config.use_grayscale else 3
    model = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=in_out_channels,
        out_channels=in_out_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",  # Cross-attention at 64x64
            "CrossAttnDownBlock2D",  # Cross-attention at 32x32
            "CrossAttnDownBlock2D",  # Cross-attention at 16x16
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",    # Cross-attention at 16x16
            "CrossAttnUpBlock2D",    # Cross-attention at 32x32
            "CrossAttnUpBlock2D",    # Cross-attention at 64x64
            "UpBlock2D",
        ),
        cross_attention_dim=config.embedding_dim, # This must match the output of our cond_embedding network
    )

    # The network to embed the concentration value (with increased dimension for better discrimination)
    cond_embedding = ConcentrationEmbedding(in_features=1, out_features=config.embedding_dim)

    # The noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    # 3. TRAINING SETUP
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cond_embedding.parameters()), lr=config.learning_rate
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # 4. START TRAINING
    train_loop(
        config,
        model,
        cond_embedding,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
        val_dataloader,
    )

    # 5. FINAL INFERENCE
    print("\n--- Training Finished ---")
    print("Running final inference on all concentrations...")
    
    # Setup device - use specified GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Load the best trained model
    best_model_path = os.path.join(config.output_dir, "best_model")
    if os.path.isdir(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        load_dir = best_model_path
    else:
        print(f"Best model not found. Loading final model from {config.output_dir}...")
        load_dir = config.output_dir

    # Load models
    model = UNet2DConditionModel.from_pretrained(load_dir).to(device)
    cond_embedding = ConcentrationEmbedding(in_features=1, out_features=128)
    
    cond_path = os.path.join(load_dir, "cond_embedding.pth")
    cond_embedding.load_state_dict(torch.load(cond_path, map_location=device, weights_only=True))
    cond_embedding = cond_embedding.to(device)
    
    generate_images(
        model,
        cond_embedding,
        noise_scheduler,
        config.all_available_concentrations,
        config.num_epochs,
        config,
        device
    )
    
    # 6. QUALITY VERIFICATION
    print("\nRunning quality verification...")
    verify_model_quality(model, cond_embedding, noise_scheduler, train_dataset, config, device)

if __name__ == "__main__":
    # Parse command line arguments for multi-GPU training
    parser = argparse.ArgumentParser(description='Train diffusion model with different configurations')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (0, 1, 2, 3)')
    parser.add_argument('--holdout', type=float, default=None, help='Concentration to hold out (e.g., 5.0)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.gpu is not None:
        config.gpu_id = args.gpu
    
    if args.holdout is not None:
        config.test_concentration = args.holdout
        config.known_concentrations = [c for c in config.all_available_concentrations if c != config.test_concentration]
        config.unknown_concentrations_for_generation = [config.test_concentration]
        base_output = os.path.join(REPO_ROOT, "results", "Generation")
        config.output_dir = os.path.join(base_output, f"holdout_{str(config.test_concentration).replace('.', '_')}")
        values = torch.tensor(config.known_concentrations, dtype=torch.float32)
        config.concentration_mean = values.mean().item()
        std = values.std(unbiased=False).item()
        config.concentration_std = std if std > 0 else 1.0
    
    if args.epochs is not None:
        config.num_epochs = args.epochs
    
    # Print final configuration to verify
    print(f"\n{'='*60}")
    print(f"FINAL CONFIGURATION (after command line override)")
    print(f"{'='*60}")
    print(f"GPU: {config.gpu_id}")
    print(f"Holdout (test) concentration: {config.test_concentration}")
    print(f"Training concentrations: {config.known_concentrations}")
    print(f"Generation concentrations: {config.unknown_concentrations_for_generation}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Output dir: {config.output_dir}")
    print(f"{'='*60}\n")
    
    main()
