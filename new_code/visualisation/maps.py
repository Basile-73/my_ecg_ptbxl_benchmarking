COLOR_MAP = {
    'noisy_input': '#808080',  # Grey (baseline)
    'ant_drnn': '#ffbb78',
    'chiang_dae': '#ff7f0e',
    'fcn': '#aec7e8',         # Light blue (Stage1)
    'drnet_fcn': '#1f77b4',   # Dark blue (Stage2)
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
    'mecge_phase': '#f6c453',      # warm golden yellow
    'mecge_phase_250': '#e0a800',  # deeper amber
    # 'imunet_mamba_bn': '#ff7f0e',  # Orange
    # 'imunet_mamba_bottleneck': '#1C8AC9',  # Cyan-blue
    # 'imunet_mamba_up': '#17becf',  # Cyan/Teal
    # 'imunet_mamba_early': '#391CC9', # Purple-blue
    # 'imunet_mamba_late': '#bcbd22',  # Yellow-green
    'mamba1_3blocks': '#8ecae6',      # light blue
    'drnet_mamba1_3blocks': '#005f73',# dark blue
    'mamba2_3blocks': '#c77dff',        # light purple
    'drnet_mamba2_3blocks': '#5a189a',  # dark purple
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
    'mecge_phase': 'MECGE-Phase',
    'mecge_phase_250': 'MECGE-Phase-250',
    'mecge': 'MECGE',
    'mamba1_3blocks': 'UNet Mamba1-3B',
    'drnet_mamba1_3blocks': 'DRNET Mamba1-3B',
    'mamba2_3blocks': 'UNet Mamba2-3B',
    'drnet_mamba2_3blocks': 'DRNET Mamba2-3B',
}

OUR_MODELS = [
    'mamba1_3blocks',
    'drnet_mamba1_3blocks',
    'mamba2_3blocks',
    'drnet_mamba2_3blocks',
]
