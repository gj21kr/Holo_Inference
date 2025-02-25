import torch
import numpy as np
import scipy

class Resampler:
    def __init__(
            self, interp_mode, original_spacing, 
            target_spacing, original_shape, target_shape=None, 
            out_dtype='tensor', with_channels=False, device='cuda'
        ):
        self.interp_mode = interp_mode
        self.original_spacing = original_spacing
        self.target_spacing = target_spacing
        self.original_shape = original_shape
        self.target_shape = target_shape
        self.out_dtype=out_dtype
        self.with_channels=with_channels
        self.device=device

    def __call__(self, raw, revert=False, out_dtype=None, with_channels=None, device=None):
        if out_dtype is None: out_dtype=self.out_dtype
        if with_channels is None: with_channels=self.with_channels
        if device is None: device=self.device
        
        if isinstance(raw, dict):
            raw = raw['image']
            make_dict = True
        else:
            make_dict = False

        if revert==True:
            mode = self.interp_mode.split('_')[-1]
            original_shape = self.target_shape
            original_spacing = self.target_spacing
            target_shape = self.original_shape
            target_spacing = self.original_spacing
        else:
            mode = self.interp_mode.split('_')[0]
            original_shape = self.original_shape
            original_spacing = self.original_spacing
            target_spacing = self.target_spacing
            target_shape = self.target_shape if self.target_shape is not None else list(np.rint(
                np.array(original_shape) * np.array(original_spacing) / np.array(target_spacing)).astype(int))
            if self.target_shape is None: self.target_shape = target_shape
        if mode=='nearest': mode='nearest-exact'
        target_shape = [int(s) for s in target_shape]
        
        if raw is None:
            return None

        if mode.isnumeric():
            if torch.is_tensor(raw):
                raw = raw.cpu().numpy()
            zoom = [t / o for t, o in zip(target_shape, original_shape)]
            to_bool = True if len(np.unique(raw)) == 2 else False
            if raw.ndim == 4:
                x = np.empty((raw.shape[0], *target_shape))
                for c in range(raw.shape[0]):
                    x[c] = scipy.ndimage.zoom(
                        raw[c], zoom, order=int(mode), mode='constant', cval=np.min(raw))
            elif raw.ndim == 3:
                x = scipy.ndimage.zoom(
                    raw, zoom, order=int(mode), mode='constant', cval=np.min(raw))
            if to_bool:
                x = (x > 0.5).astype(np.uint8)
            if out_dtype == 'tensor':
                return torch.from_numpy(x).type(torch.FloatTensor).to(device)
            else:
                return x

        if not torch.is_tensor(raw):
            x = torch.from_numpy(raw.copy())
        else:
            x = raw.clone().detach()

        if x.ndim == 3:
            channel_in = 1
            with_channels = False
        elif x.ndim == 4:
            channel_in = x.shape[0]
            with_channels = True
            
        x = torch.reshape(x.type(torch.FloatTensor),(1, channel_in, *x.shape[-3:])).to(device)
        x = torch.nn.functional.interpolate(
            x, size=target_shape, mode=mode, align_corners=True if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None)

        if device=='cpu': x = x.cpu()
        if out_dtype == 'tensor':
            x = x[0, 0] if not with_channels else x[0]
        else:
            x = x.numpy()[0, 0] if not with_channels else x.numpy()[0]
        if make_dict:
            x = {"image":x}
        return x