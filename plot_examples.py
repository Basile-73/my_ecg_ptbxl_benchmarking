from mycode.classification.utils.utils import load_dataset

X, y = load_dataset('data/physionet.org/files/ptb-xl/1.0.3/', 100)

################################################################################
# Counting
################################################################################
X_train = X[y['strat_fold'] <= 8]
X_val = X[y['strat_fold'] == 9]
X_test = X[y['strat_fold'] == 10]
print(f'Training set size: {X_train.shape[0]}')
print(f'Validation set size: {X_val.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

################################################################################
# Plotting
################################################################################

# plot the first 12 lead ecg
example = X[0]
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(4, 1, figsize=(7.5, 4*8/6))
x_values = np.arange(example.shape[0]) / 100

# Plot first 2 leads (leads 7, 8)
for i, lead_idx in enumerate([6, 7]):
    axs[i].plot(x_values[:500], example[:500,lead_idx], color='green', linewidth=4)
    axs[i].set_yticklabels([])
    axs[i].set_xticks([])

# Add separator indicating missing channels
axs[2].text(0.5, 0.5, '⋮', ha='center', va='center', fontsize=30, transform=axs[2].transAxes)
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)
axs[2].axis('off')

# Plot last lead (lead 12)
axs[3].plot(x_values[:500], example[:500,11], color='green', linewidth=4)
axs[3].set_yticklabels([])
axs[3].set_xticks([])

plt.tight_layout()
plt.show()
# save figure to plot_examples/ecg_example_12.png
fig.savefig('plot_examples/ecg_example_12.png', dpi=300)

# Also save blue version of multi-lead plot
fig_blue, axs_blue = plt.subplots(4, 1, figsize=(7.5, 4*8/6))

# Plot first 2 leads (leads 7, 8)
for i, lead_idx in enumerate([6, 7]):
    axs_blue[i].plot(x_values[:500], example[:500,lead_idx], color='blue', linewidth=4)
    axs_blue[i].set_yticklabels([])
    axs_blue[i].set_xticks([])

# Add separator indicating missing channels
axs_blue[2].text(0.5, 0.5, '⋮', ha='center', va='center', fontsize=30, transform=axs_blue[2].transAxes)
axs_blue[2].set_xlim(0, 1)
axs_blue[2].set_ylim(0, 1)
axs_blue[2].axis('off')

# Plot last lead (lead 12)
axs_blue[3].plot(x_values[:500], example[:500,11], color='blue', linewidth=4)
axs_blue[3].set_yticklabels([])
axs_blue[3].set_xticks([])

plt.tight_layout()
fig_blue.savefig('plot_examples/ecg_example_12_blue.png', dpi=300)
plt.close(fig_blue)

# Create separate plots for lead 7 and lead 8
for lead_idx in [6, 7, 8]:  # 0-indexed, so 6=lead7, 7=lead8
    fig_single, ax = plt.subplots(1, 1, figsize=(7.5, 8/6))
    ax.plot(x_values[:500], example[:500, lead_idx], color='green', linewidth=4)
    ax.set_yticklabels([])
    ax.set_xticks([])
    plt.tight_layout()
    fig_single.savefig(f'plot_examples/ecg_lead_{lead_idx+1}.png', dpi=300)
    plt.close(fig_single)

    # Also save blue version of clean data
    fig_single_blue, ax_blue = plt.subplots(1, 1, figsize=(7.5, 8/6))
    ax_blue.plot(x_values[:500], example[:500, lead_idx], color='blue', linewidth=4)
    ax_blue.set_yticklabels([])
    ax_blue.set_xticks([])
    plt.tight_layout()
    fig_single_blue.savefig(f'plot_examples/ecg_lead_{lead_idx+1}_blue.png', dpi=300)
    plt.close(fig_single_blue)

from ecg_noise_factory.noise import NoiseFactory

nf = NoiseFactory(
    'noise/data',
    100,
    'noise/configs/strong.yaml'
)

# reshape example to (1, 1000, 12)
example_reshaped = example[np.newaxis, :, :]
noisy_ecg = nf.add_noise(example_reshaped, 0, 2,1)

nf_light = NoiseFactory(
    'noise/data',
    100,
    'noise/configs/light.yaml'
)

noisy_ecg_light = nf_light.add_noise(example_reshaped, 0, 2,1)

# plot lead 8 of noisy ecg (color red)
fig_noisy, ax_noisy = plt.subplots(1, 1, figsize=(7.5, 8/6))
ax_noisy.plot(x_values[:500], noisy_ecg[0,:500,7], color='red', linewidth=4)
ax_noisy.set_yticklabels([])
ax_noisy.set_xticks([])
plt.tight_layout()
fig_noisy.savefig('plot_examples/ecg_lead_8_noisy.png', dpi=300)
plt.close(fig_noisy)

from new_code.models.UNet.Stage1_UNet import UNet
from new_code.models.UNet.Stage1_UNet_Mamba_Block import UNetMambaBlock
mamba_unet = UNetMambaBlock(in_channels=1, input_length=1000, n_blocks=3)


weights_path_mamba = '/local/home/bamorel/my_ecg_ptbxl_benchmarking/new_code/model_weights/NEXT_mamba_ptb_xl_ptb_xl_best_3600_mamba1_3blocks_ptb.pth'
import torch
mamba_unet.load_state_dict(torch.load(weights_path_mamba, map_location=torch.device('cuda')))
mamba_unet.cuda()
mamba_unet.eval()
# pass lead 8 noisy ecg through unet
with torch.no_grad():
    input_tensor = torch.tensor(noisy_ecg_light[:, :, 7:8], dtype=torch.float32).permute(0, 2, 1)  # (B, C, L)
    # reshape to (B, 1, C, L)
    input_tensor = input_tensor.unsqueeze(1).cuda()  # (B, 1, C, L)
    denoised_output = mamba_unet(input_tensor)
    # pass through mamba unet
    denoised_output = denoised_output.squeeze(1).permute(0, 2, 1).cpu().numpy()  # (B, L, C)

# plot lead 8 of denoised ecg (color blue)
fig_denoised, ax_denoised = plt.subplots(1, 1, figsize=(7.5, 8/6))
ax_denoised.plot(x_values[:500], denoised_output[0,:500,0], color='blue', linewidth=4)
ax_denoised.set_yticklabels([])
ax_denoised.set_xticks([])
plt.tight_layout()
fig_denoised.savefig('plot_examples/ecg_lead_8_denoised.png', dpi=300)
plt.close(fig_denoised)

# Denoise all relevant leads (7, 8, 9, 11, 12)
denoised_leads = {}
for lead_idx in [6, 7, 8, 10, 11]:  # 0-indexed
    with torch.no_grad():
        input_tensor = torch.tensor(noisy_ecg_light[:, :, lead_idx:lead_idx+1], dtype=torch.float32).permute(0, 2, 1)
        input_tensor = input_tensor.unsqueeze(1).cuda()
        output = mamba_unet(input_tensor)
        denoised_leads[lead_idx] = output.squeeze(1).permute(0, 2, 1).cpu().numpy()[0, :, 0]

# Create multi-lead plot with denoised signals
fig_multi, axs_multi = plt.subplots(4, 1, figsize=(7.5, 4*8/6))

# Plot first 2 leads (leads 7, 8)
for i, lead_idx in enumerate([6, 7]):
    axs_multi[i].plot(x_values[:500], denoised_leads[lead_idx][:500], color='blue', linewidth=4)
    axs_multi[i].set_yticklabels([])
    axs_multi[i].set_xticks([])

# Add separator indicating missing channels
axs_multi[2].text(0.5, 0.5, '⋮', ha='center', va='center', fontsize=30, transform=axs_multi[2].transAxes)
axs_multi[2].set_xlim(0, 1)
axs_multi[2].set_ylim(0, 1)
axs_multi[2].axis('off')

# Plot last lead (lead 12)
axs_multi[3].plot(x_values[:500], denoised_leads[11][:500], color='blue', linewidth=4)
axs_multi[3].set_yticklabels([])
axs_multi[3].set_xticks([])

plt.tight_layout()
fig_multi.savefig('plot_examples/ecg_denoised_multi.png', dpi=300)
plt.close(fig_multi)
