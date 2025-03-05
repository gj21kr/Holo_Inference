from __future__ import annotations
import os
import gc
from joblib import Parallel, delayed
import torch
import numpy as np
from monai.transforms import Transform, MapTransform, LoadImage
from monai.utils.enums import TransformBackends
from monai.networks.layers import GaussianFilter
from monai.config import KeysCollection, NdarrayTensor

from transforms.utils import optional_import, convert_data_type, convert_to_cupy

cpx, has_cupyx = optional_import("cupyx")
ci, has_ci = optional_import("cucim")
cupyimg, has_cupyimg = optional_import("cupyimg")
has_cupyx, has_ci, has_cupyimg = False, False, False

job_threshold = 10
__all__ = [
    "Normalization", "Normalizationd",
    "ZNormalization", "ZNormalizationd",
    "MaskFilter", "ForegroundFilter",
    "BinaryErosion", "BinaryDilation",
    "RemoveSamllObjects", "RemoveDistantObjects",
    "GaussianSmoothing", "KeepLargestComponent",
    "BinaryFillHoles", "ConnectComponents",
    "Threshold", "HU_filter", "BoneMaskedArrayd",
    "TargetMultiWindowingd", "TargetMultiWindowing",
    "DuplicateChanneld", 
    "RandMultiWindowingd", "RandMultiWindowing",
    "Antialiasingd"
]

def _to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

# from https://github.com/Project-MONAI/MONAI/issues/3178
class Antialiasingd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        sigma: Union[Sequence[float], float] = 1.0,
        approx: str = "erf",
        threshold: float = 0.5,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sigma = sigma
        self.approx = approx
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = _to_tensor(d[key])

            gaussian_filter = GaussianFilter(img.ndim - 1, self.sigma, approx=self.approx)

            labels = torch.unique(img)[1:]
            new_img = torch.zeros_like(img)
            for label in labels:
                label_mask = (img == label).to(torch.float)
                blurred = gaussian_filter(label_mask.unsqueeze(0)).squeeze(0)
                new_img[blurred > self.threshold] = label
            d[key] = new_img
        return d


class DuplicateChanneld(MapTransform):
    def __init__(self, keys: KeysCollection, target_channels: int):
        super().__init__(keys)
        self.target_channels = target_channels

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = torch.cat([_to_tensor(d[key])]*self.target_channels, dim=0)
        return d

class RandMultiWindowingd(MapTransform):
    def __init__(self, keys: KeysCollection, window: list, ranges: list, output_nums: int):
        self.keys = keys
        self.windowing = RandMultiWindowing(window, ranges, output_nums)
        
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            orig_shape = d[k].shape
            img = self.windowing(_to_tensor(d[key]))
            d[k] = torch.stack(img, dim=0)
            # print(f"MultiWindowing: {orig_shape} -> {d[k].shape}")
        return d

class RandMultiWindowing(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self, window:list, ranges:list, output_nums:int):
        self.window = window
        self.ranges = ranges
        self.znorm = ZNormalization(contrast=window)

        if output_nums % 2 == 0:
            print('Warning Message from transforms.utils.custom_intensity.py RandMultiWindowing',
                '\n\tThe output_nums should be odd number. The output_nums is changed to', output_nums+1)
            output_nums += 1
        elif output_nums == 1:
            print('Warning Message from transforms.utils.custom_intensity.py RandMultiWindowing',
                '\n\tThe output_nums should be greater than 1. The output_nums is changed to 3')
            output_nums = 3
        self.output_nums = output_nums
    def __call__(self, img):
        img = _to_tensor(img)
        nums = [0]+list(np.random.randint(self.ranges[0], self.ranges[1], self.output_nums-1))
        return [self.znorm(img, [self.window[0]-num, self.window[1]+num]) for num in nums]

class TargetMultiWindowing(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self, window:list, ranges:list, output_nums:int):
        self.window = window
        self.znorm = ZNormalization(contrast=window)
        self.nums = [0]+list(np.random.randint(ranges[0], ranges[1], output_nums-1))
    def __call__(self, img):
        img = _to_tensor(img)
        return [self.znorm(img, [self.window[0]-num, self.window[1]+num]) for num in self.nums]

class TargetMultiWindowingd(MapTransform):
    def __init__(self, keys: KeysCollection, window: list, ranges: list, output_nums: int):
        self.keys = keys
        self.windowing = TargetMultiWindowing(window, ranges, output_nums)
        
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            orig_shape = d[k].shape
            img = self.windowing(d[k])
            d[k] = torch.stack(img, dim=0)
            # print(f"MultiWindowing: {orig_shape} -> {d[k].shape}")
        return d
    
class BoneMaskedArrayd(MapTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]
    def __init__(self, keys: KeysCollection, phase: str, min_val: int, pname: str, bone_folder=None):
        # print('The bone mask filename should be a form of pname+.nii.gz')
        self.keys = keys
        if bone_folder is None:
            self.bone_root = os.path.join('/nas124/Data_Internal_NII/masks/TotalSegmentatorV2', phase)
            # self.bone_root = os.path.join('/home/jepark/Data/NAS3/nas_124_v1_miai/Data_Internal_NII/masks/TotalSegmentatorV2', phase)
        else:
            self.bone_root = os.path.join(bone_folder, phase)
        self.min_val = min_val 
        self.pname = pname
        
    def __call__(self, data):
        d = dict(data)        
        bone_file = os.path.join(self.bone_root, self.pname, 'Bone.nii.gz')
        if not os.path.isfile(bone_file):
            print('No bone file exists!', bone_file)
            return d
        bone = (LoadImage(image_only=True,reader="ITKReader")(bone_file)>0)
        for key in self.keys:
            if d[key].shape[-3:] != bone.shape[-3:]:
                print('The shape of the bone mask is different!', self.pname, d[key].shape, bone.shape)
                return d
            if d[key].ndim==4:
                d[key][:,bone==True] = self.min_val
            else:
                d[key][bone==True] = self.min_val
                # self.savenifti(d[key], 'bonemasked')
                # self.savenifti(new_mask.numpy().astype(int), 'mask')
        return d
    def savenifti(self, image, name):
        import SimpleITK as sitk 
        if not os.path.isfile(f'/home/jepark/Train_Research/{name}.nii.gz'):
            image = sitk.GetImageFromArray(image)
            sitk.WriteImage(image, f'/home/jepark/Train_Research/{name}.nii.gz')
            
class Threshold(Transform):
    def __init__(self, threshold=[0.5]):
        if type(threshold) == list:
            self.threshold = threshold
        else:
            self.threshold = [threshold]
        self.apply_labels = None
        self.image_require = False
        import numpy as np
        self.lib = np
    def __call__(self, mask):
        def work(mask, thres_val):
            mask = self.lib.where(mask<thres_val, 0, 1)
            return mask

        if mask.ndim==4:
            if self.apply_labels==None : 
                self.apply_labels = list(range(mask.shape[0]))
                self.threshold = [0.5 for _ in range(mask.shape[0])]
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = self.apply_labels
            # m = Parallel(n_jobs=n_jobs)(
            # 	delayed(work)(mask[this_],self.threshold[this_]) for this_ in self.apply_labels
            # )
            m = []
            for this_ in self.apply_labels:
                m.append(work(mask[this_],self.threshold[this_]))
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask.astype(self.lib.uint8)
        else:
            return work(mask, self.threshold[0]).astype(self.lib.uint8)

class Normalization(Transform):
    """Subtract mean and divide by standard deviation.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self):
        pass
    def __call__(self, img, mask):
        val_min, val_max = self.cal_vals(img, mask)
        if val_std == 0:
            print('masked pixels have 0 of std!'); return img
        img -= val_min
        img /= (val_max-val_min)
        return img
    def cal_vals(self, img, mask):
        if mask.ndim==img.ndim:
            return np.min(img[mask>0]), np.max(img[mask>0])
        else:
            temp = np.zeros_like(img[0])
            temp[img[0]>0] = img[0, img[0]>0]
            return np.min(temp[temp>0]), np.max(temp[temp>0])

class Normalizationd(MapTransform):
    def __init__(self, keys:KeysCollection, image_key:str="image", label_key:str="label"):
        self.keys = keys
        self.image_key = image_key
        self.label_key = label_key
        self.transform = Normalization()

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]; mask = d[self.label_key]
        d[self.image_key] = self.transform(img, mask)
        return d

class ZNormalization(Transform):
    """Subtract mean and divide by standard deviation.
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self, contrast:list=None):
        self.contrast = contrast


    def __call__(self, img, new_contrast=None):    
        if new_contrast is not None:
            self.contrast = new_contrast    
        img = _to_tensor(img).float()
        img = torch.clamp(img, min=self.contrast[0], max=self.contrast[1])

        mean_ = torch.mean(img)
        std_ = torch.std(img)

        if std_ == 0:
            print('masked pixels have 0 of std!'); return img
            
        img = torch.div(torch.add(img, -mean_), std_)
        img = torch.mul(torch.add(img, 1), 0.5)
        return img 

class ZNormalizationd(MapTransform):
    def __init__(self, keys:KeysCollection, contrast:list):
        self.keys = keys
        self.transform = ZNormalization(contrast)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = self.transform(_to_tensor(d[k]))

        return d

class MaskFilter:
    def __init__(self, mask_name=None, mask_path=None, apply_labels=None, mode=1):
        import os
        import numpy as np
        import SimpleITK as sitk
        self.lib = np
        self.mode= mode
        self.apply_labels = apply_labels		
        if mask_name is not None and mask_path is not None:
            self.image_require = False
            if not '.nii.gz' in mask_name: mask_name = mask_name+'.nii.gz'
            self.ref = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, mask_name)))
        else:
            self.image_require = 'mask'

    def __call__(self, mask, image=None):
        def work(m,i,mode=1):
            m_ = m.copy()
            m_[i<1] = 0
            return m_ 
        if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
        if len(self.apply_labels)>job_threshold:
            n_jobs = job_threshold
        else:
            n_jobs = self.apply_labels
        if self.image_require!=False:
            self.ref = image.copy()
        m = Parallel(n_jobs=n_jobs)(
            delayed(work)(mask[this_], self.ref, self.mode) for this_ in self.apply_labels
        )
        if mask.shape[0]==len(self.apply_labels):
            return self.lib.stack(m,axis=0).astype(self.lib.uint8)
        else:
            for i,c in enumerate(self.apply_labels):
                mask[c] = m[i]
            return mask.astype(self.lib.uint8)


class ForegroundFilter:
    def __init__(self, threshold, apply_labels=None):
        self.threshold = threshold
        self.apply_labels = apply_labels
        self.image_require = True
        import numpy as np
        self.lib = np
    def __call__(self, mask, image):
        def work(m,i,thres):
            m_ = m.copy()
            m_[i<thres[0]] = 0
            m_[i>thres[1]] = 0
            return m_.astype(m.dtype)
        if mask.ndim==4:
            if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = self.apply_labels
            m = Parallel(n_jobs=n_jobs)(
                delayed(work)(mask[this_], image, self.threshold) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask			
        else:
            return work(mask, image, self.threshold)

class BinaryErosion:
    def __init__(self, connectivity=3, iterations=1, apply_labels=None):
        import scipy ; import numpy as np
        self.structure = scipy.ndimage.generate_binary_structure(3, connectivity) 
        self.apply_labels = apply_labels
        self.image_require = False
        self.iterations = iterations
        self.work = scipy.ndimage 
        self.lib = np
    def __call__(self, mask):
        if mask.ndim==4:			
            if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = self.apply_labels
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work.binary_erosion)(
                    mask[this_], structure=self.structure, iterations=self.iterations) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask			
        else:
            mask = self.work.binary_erosion(mask, structure=self.structure, iterations=self.iterations)
            return mask.astype(self.lib.uint8)	

    
class BinaryDilation:
    def __init__(self, connectivity=3, iterations=1, apply_labels=None):
        import scipy ; import numpy as np
        self.structure = scipy.ndimage.generate_binary_structure(3, connectivity) 
        self.apply_labels = apply_labels
        self.image_require = False
        self.iterations = iterations
        self.work = scipy.ndimage
        self.lib = np
    def __call__(self, mask):
        if mask.ndim==4:
            if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work.binary_dilation)(
                    mask[this_], structure=self.structure, iterations=self.iterations) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask			
        else:
            mask = self.work.binary_dilation(mask, structure=self.structure, iterations=self.iterations)
            return mask.astype(self.lib.uint8)	

class RemoveSamllObjects:
    def __init__(self, min_size=1e03, connectivity=3, apply_labels=None):
        import skimage
        self.morphology = skimage.morphology
        self.connectivity = connectivity
        self.minSz = min_size
        self.apply_labels = apply_labels
        self.image_require = False
        import numpy as np
        self.lib = np

    def __call__(self, mask):
        if has_cupyimg:
            m = convert_to_cupy(m)
        if type(mask)==list: mask = self.lib.array(mask)

        def work(m):
            m = self.morphology.remove_small_objects(
                m.astype(bool), min_size=self.minSz, connectivity=self.connectivity)
            return m
        if mask.ndim==4:
            if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
            
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(work)(mask[this_]) for this_ in self.apply_labels
            )		
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask.astype(self.lib.uint8)
        else:
            return work(mask).astype(self.lib.uint8)	
            

class RemoveDistantObjects:
    def __init__(self, struct=3, connectivity=3, dim=3, distance=100, image_center=False, apply_labels=None):
        self.connectivity = connectivity
        self.apply_labels = apply_labels
        self.distance = distance//2
        self.image_center = image_center
        self.image_require = False
        if has_cupyimg==True:
            self.ndi = cupyimg.scipy.ndimage
            self.lib = cupyimg
        else:        
            import scipy as sp
            import numpy as np
            self.ndi = sp.ndimage
            self.lib = np
        if type(struct)==int and dim==3:
            self.struct = self.lib.ones((struct,struct,struct))
        elif type(struct)==int and dim==2:
            self.struct = self.lib.ones((struct,struct))
        if has_cupyimg:
            self.struct = convert_to_cupy(struct)

    def work(self, m):
        if has_cupyimg:
            m = convert_to_cupy(m) 
            
        footprint = self.ndi.generate_binary_structure(m.ndim, self.connectivity)
        features, num_features = self.ndi.label(m, footprint)
        if self.image_center==True:
            img_com = self.lib.array(m.shape[-3:])//2
            t = self.lib.zeros_like(m)
            t[
                img_com[0]-self.distance:img_com[0]+self.distance,
                img_com[1]-self.distance:img_com[1]+self.distance,
                img_com[2]-self.distance:img_com[2]+self.distance
            ] = m[
                img_com[0]-self.distance:img_com[0]+self.distance,
                img_com[1]-self.distance:img_com[1]+self.distance,
                img_com[2]-self.distance:img_com[2]+self.distance
            ]
            m = t; del t; gc.collect()
        else:
            img_com = self.lib.array(self.ndi.center_of_mass(m, features))
            coms = self.ndi.center_of_mass(
                m, features, self.lib.arange(1,num_features+1))
            temp = self.lib.zeros_like(m)
            for indx, com in enumerate(coms):
                dist = self.lib.linalg.norm(self.lib.array(com)-img_com) 
                if dist < self.distance:
                    temp[features==indx+1] = 1
            m= temp

        if has_cupyimg==True:
            m = cupyimg.asnumpy(m)
        return (m>0)

    def __call__(self, mask):
        mask = self.lib.array(mask)
        if mask.ndim==4:
            if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work)(mask[this_]) for this_ in self.apply_labels
            )		
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask.astype(self.lib.uint8)		
        else:
            return self.work(mask).astype(self.lib.uint8)		
            

class GaussianSmoothing:
    def __init__(self, sigma=1, mode='nearest', channel_axis=None):
        if has_ci:
            from cucim.skimage.filters import gaussian
            import cupy as cp
            self.filter = gaussian
            self.lib = cp
        else:
            from skimage.filters import gaussian
            import numpy as np
            self.filter = gaussian
            self.lib = np
        self.sigma = sigma
        self.mode = mode
        self.channel_axis = channel_axis
        self.image_require = False

    def __call__(self, mask):
        output = self.filter(
            mask, sigma=self.sigma, mode=self.mode, channel_axis=self.channel_axis)
        return (output>0).astype(mask.dtype)

class KeepLargestComponent:
    def __init__(self, connectivity=3, apply_labels=None):
        self.connectivity = connectivity
        self.apply_labels = apply_labels
        self.image_require = False
        if has_ci:
            from scipy.ndimage import label, generate_binary_structure, find_objects
            import cupy as cp
            self.label = label
            self.structure = generate_binary_structure
            self.find_objects = find_objects
            self.lib = cp
        else:
            from scipy.ndimage import label, generate_binary_structure, find_objects
            import numpy as np
            self.label = label
            self.structure = generate_binary_structure
            self.find_objects = find_objects
            self.lib = np
        
    def work(self,mask):
        mask = self.lib.array(mask)
        result = self.lib.zeros_like(mask)
        if has_ci : 
            mask = convert_to_cupy(mask)
        # Define the structure for connected component analysis
        struct = self.structure(3, self.connectivity)
        labeled_arr, num_components = self.label((mask>0), struct)

        if num_components <= 1:
            if has_ci : mask = self.lib.asnumpy(mask)
            return (mask>0).astype(self.lib.uint8)
        else:
            # Find the indices and sizes of each component
            component_sizes = [self.lib.sum(labeled_arr == i) for i in range(1, num_components + 1)]
            largest_component_index = self.lib.argmax(component_sizes) + 1
            # Filter out all components except the largest one
            largest_component_slice = self.find_objects(labeled_arr == largest_component_index)[0]
            # mask[largest_component_slice] *= 1
            # mask[labeled_arr != largest_component_index] *= 0
            # return (mask>0).astype(self.lib.uint8)
            result[largest_component_slice] = 1
            return result.astype(self.lib.uint8)

    def __call__(self, mask): 
        mask = self.lib.array(mask)
        if mask.ndim == 4:
            if self.apply_labels==None : self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work)(mask[this_]) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return self.work(mask)
        else:
            return self.work(mask)

class BinaryFillHoles:
    def __init__(self, structure=None, dim=3, apply_labels=None):
        if has_cupyx:
            import cupyx.scipy.ndimage.morphology as morph
            import cupy as cp
            self.morph = morph
            self.lib = cp
        else: 
            import scipy.ndimage.morphology as morph 
            import numpy as np
            self.morph = morph
            self.lib = np
        if type(structure) == int:
            if dim==3:
                structure = self.lib.ones((structure,structure,structure))
            elif dim==2:
                structure = self.lib.ones((structure,structure))
        self.structure = structure
        self.apply_labels = apply_labels
        self.image_require = False

    def work(self, mask):
        if has_cupyx: mask = convert_to_cupy(mask)
        mask = self.morph.binary_fill_holes(mask, structure=self.structure).astype(mask.dtype)
        if has_cupyx: mask = self.lib.asnumpy(mask)
        return (mask>0)
    def __call__(self, mask):        
        if mask.ndim==4:
            if self.apply_labels==None: self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work)(mask[this_]) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return self.work(mask).astype(self.lib.uint8)
        else:
            return self.work(mask).astype(self.lib.uint8)

class ConnectComponents:
    def __init__(self, connectivity=26, iterations=1, apply_labels=None):
        # connectivity : 26, 18, and 6 (3D) are allowed
        import cc3d
        import numpy as np
        if connectivity == 1: 
            connectivity = 6
        elif connectivity == 2:
            connectivity = 18
        elif connectivity == 3:
            connectivity = 26
        if connectivity not in [26,18,6]:
            print(f"ConnectComponents - connectivity : 26, 18, and 6 (3D) are allowed. {connectivity} is changed to 26")
            connectivity = 26

        self.connectivity = connectivity 
        self.apply_labels = apply_labels
        self.image_require = False
        self.work = cc3d; self.lib = np
    def __call__(self, mask):		
        if mask.ndim==4:
            if self.apply_labels==None: self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = self.apply_labels
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work.connected_components)(mask[this_], connectivity=self.connectivity) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask.astype(self.lib.uint8)
        else:
            mask = self.work.connected_components(mask, connectivity=self.connectivity)
        return (mask>0).astype(self.lib.uint8)

class HU_filter:
    def __init__(self, percentage=0.5, mode='up', fnc='max', apply_labels=None):
        import numpy as np
        from scipy import ndimage
        self.apply_labels = apply_labels
        self.image_require = True
        self.percentage = percentage
        self.mode = mode
        self.fnc = fnc
        self.lib = np
    def work(self, mask, image):
        image = image*(mask>0).astype(self.lib.uint8)
        if self.fnc == 'max':
            hu_ = self.lib.max(image)
        elif self.fnc == 'min':
            hu_ = self.lib.min(image)
        elif self.fnc == 'mean':
            hu_ = self.lib.mean(image)
        if self.mode in ['up', 'between']:
            mask[image<hu_*self.percentage] = 0
        elif self.mode in ['low', 'between']:
            mask[image>hu_*self.percentage] = 0
        return mask        
    def __call__(self, mask, image):
        if mask.ndim==4:
            if self.apply_labels==None: self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work)(mask[this_], image) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask.astype(self.lib.uint8)
        else:
            mask = self.work(mask, image)
            return mask.astype(self.lib.uint8)
        
class ConnectComponentsJ:
    def __init__(self, connectivity=1, iterations=1, apply_labels=None):
        import numpy as np
        from scipy import ndimage
        self.apply_labels = apply_labels
        self.iterations = iterations
        self.structure = ndimage.generate_binary_structure(3, connectivity)
        self.image_require = False
        self.work = ndimage.binary_closing
        self.lib = np
    def __call__(self, mask):
        if mask.ndim==4:
            if self.apply_labels==None: self.apply_labels = list(range(mask.shape[0]))
            if len(self.apply_labels)>job_threshold:
                n_jobs = job_threshold
            else:
                n_jobs = len(self.apply_labels)
            m = Parallel(n_jobs=n_jobs)(
                delayed(self.work)(
                    mask[this_], structure=self.structure, iterations=self.iterations) for this_ in self.apply_labels
            )
            if mask.shape[0]==len(self.apply_labels):
                return self.lib.stack(m,axis=0).astype(self.lib.uint8)
            else:
                for i,c in enumerate(self.apply_labels):
                    mask[c] = m[i]
                return mask.astype(self.lib.uint8)

        else:
            mask = self.work(mask, structure=self.structure, iterations=self.iterations)
            return mask.astype(self.lib.uint8)
