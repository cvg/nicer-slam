from distutils.command.build import build
import os
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

Path('./tmp_build/').mkdir(parents=True, exist_ok=True)

_src_path = os.path.dirname(os.path.abspath(__file__))
device=torch.cuda.get_device_name()
if 'V100' in device:
    device='V100'
elif 'Quadro' in device:
    device='Quadro'
elif 'A100' in device:
    device='A100'
elif '2080' in device:
    device='2080'
elif 'TITAN' in device:
    device='TITAN'
elif '3080' in device:
    device='3080'
elif '3090' in device:
    device='3090'
elif '1080' in device:
    device='1080'
build_directory=f'./tmp_build_{device}/'
print('build_directory', build_directory)
os.makedirs(build_directory, exist_ok=True)
_backend = load(name=f'_hash_encoder_{device}',
                extra_cflags=['-O3', '-std=c++14'],
                extra_cuda_cflags=[
                    '-O3', '-std=c++14', '-allow-unsupported-compiler',
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                ],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'hashencoder.cu',
                    'bindings.cpp',
                ]],
                build_directory=build_directory,
                verbose=True,
                )

__all__ = ['_backend']