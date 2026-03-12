COLOR_MAP = {
    'noisy_input': '#808080',  # Grey (baseline)
    'chiang_dae': '#ff7f0e',
    'ant_drnn': '#ffbb78',
    'fcn': '#aec7e8',         # Light blue (Stage1)
    'drnet_fcn': '#1f77b4',   # Dark blue (Stage2)
    'mecge_phase': '#f6c453',      # warm golden yellow
    'mecge': '#f6c453',
    'mecge_phase_250': '#e0a800',  # deeper amber
    'unet': '#ff9896',
    # 'unet_mamba': '#cc7675',
    # 'unet_mamba - unet': '#cc7675',
    # 'unet_mamba_bidir': '#995555',
    # 'unet_mamba_bidir - unet': '#995555',
    'drnet_unet': '#d62728',  # Dark red (Stage2)
    'imunet': '#98df8a',
    'imunet_mamba': '#6cbf5c',
    'imunet_mamba_bidir': '#3d7f3d',
    'drnet_imunet': '#2ca02c', # Dark green (Stage2)
    'imunet_origin': '#9467bd',    # Purple
    # 'imunet_mamba_bn': '#ff7f0e',  # Orange
    # 'imunet_mamba_bottleneck': '#1C8AC9',  # Cyan-blue
    # 'imunet_mamba_up': '#17becf',  # Cyan/Teal
    # 'imunet_mamba_early': '#391CC9', # Purple-blue
    # 'imunet_mamba_late': '#bcbd22',  # Yellow-green
    'mamba1_3blocks': '#8ecae6',      # light blue
    'drnet_mamba1_3blocks': '#005f73',# dark blue
    'mamba1_3blocks_ls': '#c77dff',      # light blue
    'drnet_mamba1_3blocks_ls': '#5a189a',# dark blue
    'mamba2_3blocks': '#c77dff',        # light purple
    'drnet_mamba2_3blocks': '#5a189a',  # dark purple

    'mecge_lead_1': '#f6c453',

    'unet_lead_1': '#ff9896',
    'drnet_unet_lead_1': '#d62728',  # Dark red (Stage2)
    'imunet_lead_1': '#98df8a',
    'drnet_imunet_lead_1': '#2ca02c', # Dark green (Stage2)

    'mamba1_3blocks_lead_1': '#8ecae6',      # light blue
    'drnet_mamba1_3blocks_lead_1': '#005f73',# dark blue
    'mamba2_3blocks_lead_1': '#c77dff',        # light purple
    'drnet_mamba2_3blocks_lead_1': '#5a189a',  # dark purple

    'drnet_mamba1_3blocks_l0': '#005f73',
    'drnet_mamba1_3blocks_l1': '#005f73',
    'drnet_mamba1_3blocks_l2': '#005f73',
    'drnet_mamba1_3blocks_l3': '#005f73',
    'drnet_mamba1_3blocks_l4': '#005f73',
    'drnet_mamba1_3blocks_l5': '#005f73',
    'drnet_mamba1_3blocks_l6': '#005f73',
    'drnet_mamba1_3blocks_l7': '#005f73',
    'drnet_mamba1_3blocks_l8': '#005f73',
    'drnet_mamba1_3blocks_l9': '#005f73',
    'drnet_mamba1_3blocks_l10': '#005f73',
    'drnet_mamba1_3blocks_l11': '#005f73',
    'drnet_mamba1_3blocks_all': '#005f73',
    'mamba1_3blocks_ptb_l0': '#8ecae6',
    'mamba1_3blocks_ptb_l1': '#8ecae6',
    'mamba1_3blocks_ptb_l2': '#8ecae6',
    'mamba1_3blocks_ptb_l3': '#8ecae6',
    'mamba1_3blocks_ptb_l4': '#8ecae6',
    'mamba1_3blocks_ptb_l5': '#8ecae6',
    'mamba1_3blocks_ptb_l6': '#8ecae6',
    'mamba1_3blocks_ptb_l7': '#8ecae6',
    'mamba1_3blocks_ptb_l8': '#8ecae6',
    'mamba1_3blocks_ptb_l9': '#8ecae6',
    'mamba1_3blocks_ptb_l10': '#8ecae6',
    'mamba1_3blocks_ptb_l11': '#8ecae6',
    'mamba1_3blocks_ptb_all': '#8ecae6',

    'mamba1_3blocks_3_compression': '#e76f51',    # coral/orange-red
    'mamba1_3blocks_3_no_compression': '#264653',  # dark teal
}

NAME_MAP = {
    'noisy_input': 'Noisy Input',
    'ant_drnn': 'DRNN',
    'chiang_dae': 'DAE',
    'fcn': 'FCN',
    'drnet_fcn': 'DRNET-FCN',

    'unet': 'UNet',
    'drnet_unet': 'DRNET-UNet',
    'imunet': 'IMUNet',
    'drnet_imunet': 'DRNET-IMUNet',
    'unet_lead_1': 'UNet (Lead 1)',
    'drnet_unet_lead_1': 'DRNET-UNet (Lead 1)',
    'imunet_lead_1': 'IMUNet (Lead 1)',
    'drnet_imunet_lead_1': 'DRNET-IMUNet (Lead 1)',

    'mecge_phase': 'MECGE-Phase',
    'mecge_phase_250': 'MECGE-Phase-250',
    'mecge': 'MECGE',
    'mecge_lead_1': 'MECGE (Lead 1)',

    'mamba1_3blocks': 'UNet Mamba1-3B',
    'drnet_mamba1_3blocks': 'DRNET Mamba1-3B',
    'mamba2_3blocks': 'UNet Mamba2-3B',
    'drnet_mamba2_3blocks': 'DRNET Mamba2-3B',
    'mamba1_3blocks_lead_1': 'UNet Mamba1-3B (Lead 1)',
    'drnet_mamba1_3blocks_lead_1': 'DRNET Mamba1-3B (Lead 1)',
    'mamba2_3blocks_lead_1': 'UNet Mamba2-3B (Lead 1)',
    'drnet_mamba2_3blocks_lead_1': 'DRNET Mamba2-3B (Lead 1)',
    'mamba1_3blocks_ls': 'UNet Mamba1-3B (Lead aware)',      # light blue
    'drnet_mamba1_3blocks_ls': 'DRNET Mamba1-3B (Lead aware)',

    'mamba1_3blocks_3_compression': 'UNet Mamba1-3B (Compression)',
    'mamba1_3blocks_3_no_compression': 'UNet Mamba1-3B (No Compression)',
}

OUR_MODELS = [
    'mamba1_3blocks',
    'drnet_mamba1_3blocks',
    'mamba2_3blocks',
    'drnet_mamba2_3blocks',
    'mamba1_3blocks_ls',
    'drnet_mamba1_3blocks_ls',
]

EXCLUDE_MODELS = [
    'mamba2_3blocks',
    'drnet_mamba2_3blocks',
]

CLASSIFICATION_MODEL_NAMES = {
    'fastai_xresnet1d101': 'FastAI XResNet1D-101',
    'astai_resnet1d_wang': 'FastAI ResNet1D-Wang',
    'fastai_lstm': 'FastAI LSTM',
    'fastai_lstm_bidir': 'FastAI BiLSTM',
    'fastai_fcn_wang': 'FastAI FCN-Wang',
    'fastai_inception1d': 'FastAI Inception1D',
}

plot_font_sizes = {
    'title': 16,
    'value_labels': 14,
    'axis_labels': 14,
    'legend': 11,
    'ticks': 14,
}
