import enum
from math import ceil
from cachetools import cached
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 
from .backend import _backend

class _hash_encode(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        embeddings = embeddings.contiguous()
        offsets = offsets.contiguous()
        #print(inputs)
        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=inputs.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=inputs.dtype)
        else:
            dy_dx = torch.empty(1, device=inputs.device, dtype=inputs.dtype)

        _backend.hash_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, calc_grad_inputs, dy_dx)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H]
        ctx.calc_grad_inputs = calc_grad_inputs

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        #print("backward============================")
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        #_backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs)
        grad_inputs, grad_embeddings = _hash_encode_second_backward.apply(grad, inputs, embeddings, offsets, B, D, C, L, S, H, calc_grad_inputs, dy_dx)

        if calc_grad_inputs:
            return grad_inputs, grad_embeddings, None, None, None, None
        else:
            return None, grad_embeddings, None, None, None, None

import time

def get_time():
    torch.cuda.synchronize()
    return time.time()

timing = False
        
class _hash_encode_second_backward(Function):
    @staticmethod
    def forward(ctx, grad, inputs, embeddings, offsets, B, D, C, L, S, H, calc_grad_inputs, dy_dx):

        if timing: t1 = get_time()  
        grad_inputs = torch.zeros_like(inputs)
        grad_embeddings = torch.zeros_like(embeddings)
        #import pdb; pdb.set_trace()
        #print("first backward forward")
        ctx.save_for_backward(grad, inputs, embeddings, offsets, dy_dx, grad_inputs, grad_embeddings)
        ctx.dims = [B, D, C, L, S, H]
        ctx.calc_grad_inputs = calc_grad_inputs
        # print('??????')
        # exit(0)
        _backend.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs)
        
        '''
        print("grad_embeddings")
        for g in grad_embeddings:
            if g.abs().sum() > 1e-6:
                print(g)
        '''
        if timing: 
            t2 = get_time()  
            print("++++++++++++++++++++second backward forword time:", t2 - t1)
        
        
        return grad_inputs, grad_embeddings

    @staticmethod
    def backward(ctx, grad_grad_inputs, grad_grad_embeddings):
        #print("first backward backward")
        if timing: t1 = get_time()  
        #print("grad_grad_inputs", grad_grad_inputs)
        grad, inputs, embeddings,  offsets, dy_dx, grad_inputs, grad_embeddings = ctx.saved_tensors
        B, D, C, L, S, H = ctx.dims
        calc_grad_inputs = ctx.calc_grad_inputs
        
        grad_grad = torch.zeros_like(grad)
        grad2_embeddings = torch.zeros_like(embeddings)
        
        _backend.hash_encode_second_backward(grad, inputs, embeddings, offsets, 
                                             B, D, C, L, S, H, calc_grad_inputs, dy_dx, 
                                             grad_grad_inputs,
                                             grad_grad, grad2_embeddings)
        '''
        print("grad2_embeddings")
        for g in grad2_embeddings:
            if g.abs().sum() > 1e-6:
                print(g)
        '''
        if timing: 
            t2 = get_time()  
            print("===================second backward backward time:", t2 - t1)
        
        return grad_grad, None, grad2_embeddings, None, None, None, None, None, None, None, None, None


hash_encode = _hash_encode.apply


class HashEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution) ** input_dim) # limit max number
            #params_in_level = np.ceil(params_in_level / 8) * 8 # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} params={tuple(self.embeddings.shape)}"
    
    def forward_embedding(self):
        
        x = self.embeddings

        for l in range(0, self.num_layers):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 1:
                x = self.activation(x)
                #print("forward embedding ", l, x.shape)
        return x

    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + size) / (2 * size) # map to [0, 1]
        #print(inputs)        
        # print('HASH inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        # print(inputs, self.embeddings.shape, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
        # exit(0)
        outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)        
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    def torch_forward(self,inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]
        #assert self.offsets.shape[0] == 2
        inputs = (inputs + size) / (2 * size) # map to [0, 1]
        #print(inputs)        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())
        
        input_dim = inputs.shape[-1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
      
        outputs = []
        for level in range(self.offsets.shape[0] - 1):
        
            S = np.log2(self.per_level_scale)
            scale = np.exp2(level * S) * self.base_resolution - 1
            resolution = ceil(scale) + 1
            #print("scale resolution")
            #print(scale, resolution, self.base_resolution)
            
            #import pdb; pdb.set_trace()
            
            # compute_index
            pos = inputs * scale #+ 0.5
            left = pos.long()
            wb = pos - left
            wb = (wb * wb) * (3. - 2. * wb)
            wa = 1.0 - wb
            
            #import pdb; pdb.set_trace()
            if input_dim == 3:
                volume = self.embeddings[self.offsets[level]:self.offsets[level+1]].reshape(resolution, resolution, resolution, -1).permute(2, 1, 0, 3).contiguous()

                #print(wa)
                #print(wb)
                lx, ly, lz = left.unbind(-1)

                v000 = volume[lx, ly, lz]
                v001 = volume[lx, ly, lz + 1]
                v010 = volume[lx, ly + 1, lz]
                v011 = volume[lx, ly + 1, lz + 1]
                v100 = volume[lx + 1, ly, lz]
                v101 = volume[lx + 1, ly, lz + 1]
                v110 = volume[lx + 1, ly + 1, lz]
                v111 = volume[lx + 1, ly + 1, lz + 1]

                c00 = v000 * wa[:, 2:] + v001 * wb[:, 2:]
                c01 = v010 * wa[:, 2:] + v011 * wb[:, 2:]
                c10 = v100 * wa[:, 2:] + v101 * wb[:, 2:]
                c11 = v110 * wa[:, 2:] + v111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                output = c0 * wa[:, :1] + c1 * wb[:, :1]
                outputs.append(output)
            elif input_dim == 2:
                volume = self.embeddings[self.offsets[level]:self.offsets[level+1]].reshape(resolution, resolution, -1).permute(1, 0, 2).contiguous()

                #print(wa)
                #print(wb)
                lx, ly = left.unbind(-1)
                print(lx.min(),lx.max())
                c00 = volume[lx, ly]
                c01 = volume[lx, ly + 1]
                c10 = volume[lx + 1, ly]
                c11 = volume[lx + 1, ly + 1]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                output = c0 * wa[:, :1] + c1 * wb[:, :1]
                outputs.append(output)
            else:
                raise NotImplementedError

        outputs = torch.cat((outputs), dim=-1)
        #import pdb; pdb.set_trace()
        
        #outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs



######### Neural Volume : predict 4 level of volume with conv network
class ConvTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(ConvTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        self.latent_code = nn.Parameter(torch.empty(1, encodingsize))
        std = 1e-4
        self.latent_code.data.uniform_(-std, std)

        # build template convolution stack
        self.template1 = nn.Sequential(nn.Linear(self.encodingsize, 1024), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 1024, 512
        
        self.relu = nn.LeakyReLU(0.2)
            
        for i in range(int(np.log2(self.templateres)) - 1):
            conv = nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1)
            setattr(self, "conv_%d"%(i), conv)

            if i in [2, 3, 4, 5]:
                out = nn.ConvTranspose3d(outchannels, 8, 4, 2, 1)
                setattr(self, "out_%d"%(i), out)

            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels

        #for m in [self.template1, self.template2]:
        #    models.utils.initseq(m)

    def forward(self, encoding):
        x = self.template1(self.latent_code).view(-1, 1024, 1, 1, 1)
        outs = []

        for i in range(int(np.log2(self.templateres)) - 1):
            conv = getattr(self, "conv_%d"%(i))
            x = conv(x)
            x = self.relu(x)

            if i in [2, 3, 4, 5]:
                out = getattr(self, "out_%d"%(i))
                out = out(x)
                outs.append(out)

        embeddings = []
        for level in range(4):
            cur_embedding = outs[level] # b, c,D, h, w
            #print(cur_embedding.shape)
            assert cur_embedding.shape[0] == 1
            cur_embedding = cur_embedding.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, 8)
            
            embeddings.append(cur_embedding)
        embeddings = torch.cat(embeddings, dim=0)
        
        return embeddings

class MLPDecoder(nn.Module):
    def __init__(self):
        super(MLPDecoder, self).__init__()

        self.hidden_size = 256
        self.out_feature_dim = 8*8*8*8
        self.mlp = nn.Sequential(nn.Linear(32, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.out_feature_dim)
        )

        std = 1e-4
        for idx, resolution in enumerate([2, 4, 8, 16]): 
            input = nn.Parameter(torch.empty(resolution**3, 32))
            input.data.uniform_(-std, std)
            setattr(self, "inputs_%d"%(idx), input)
         

    def forward(self):
        embeddings = []
        for i , resolution in zip(range(4), [2, 4, 8,16]):
            input = getattr(self, "inputs_%d"%(i))
            x = self.mlp(input)
            x = x.reshape(resolution, resolution, resolution, 8, 8, 8, 8).permute(0, 3, 1, 4, 2, 5, 6).contiguous().reshape(-1, 8)
            embeddings.append(x)
        embeddings = torch.cat(embeddings, dim=0)        
        return embeddings

class GridDecoder(nn.Module):
    def __init__(self):
        super(GridDecoder, self).__init__()

        std = 1e-4
        n_params = 0
        level_dim = 8
        for idx, resolution in enumerate([2, 4, 8, 16]): 
            n_params += (resolution *8)**3
        self.n_params = n_params

        embedding_length = 2048
        embedding_size = 256
        out_dim = ceil(n_params * level_dim / embedding_length)
        

        input = nn.Parameter(torch.empty(embedding_length, 32))
        input.data.uniform_(-std, std)
        self.input = input

        self.hidden_size = 256
        self.out_feature_dim = out_dim
        self.mlp = nn.Sequential(nn.Linear(32, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.hidden_size, self.out_feature_dim)
        )

        '''
            #TODO using different frequency since we only has a very small grid
            multires = 20
            from models.embedder import get_embedder
            embed_fn, input_ch = get_embedder(multires, input_dims=2)
            self.embed_fn_fine = embed_fn

            X = torch.linspace(-1, 1, 32)
            Y = torch.linspace(-1, 1, embedding_length // 32)

            xx, yy = torch.meshgrid(X, Y)

            inputs = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1)
        
            inputs = self.embed_fn_fine(inputs)
            self.register_buffer('embeddings', inputs)
            lin_dims[0] = input_ch
        '''

    def forward(self):
        embeddings = self.mlp(self.input)
        embeddings = embeddings.reshape(self.n_params, 8)
        return embeddings


class NeuralVolume(nn.Module):
    def __init__(self, input_dim=3, num_levels=4, level_dim=8, per_level_scale=2, base_resolution=16, log2_hashmap_size=24, desired_resolution=None):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution) ** input_dim) # limit max number
            print("resolution", i, resolution, params_in_level, self.max_params)
            #params_in_level = np.ceil(params_in_level / 8) * 8 # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        volume_type = "griddecoder"
        if volume_type == "NeuralVolume":
            self.volume_decoder = ConvTemplate()
        elif volume_type == "mlpdecoder":
            self.volume_decoder = MLPDecoder()
        elif volume_type == "griddecoder":
            self.volume_decoder = GridDecoder()
        else:
            raise NotImplementedError

        self.network_code = None
        for p in self.parameters():
            print(p.shape)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} params={tuple(self.embeddings.shape)}"
    
    def forward_embedding(self):
        
        embeddings = self.volume_decoder()

        return embeddings

    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + size) / (2 * size) # map to [0, 1]
        #print(inputs)        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        #TODO cache
        if inputs.requires_grad:
            network_code = self.forward_embedding().reshape(-1, self.level_dim)
            self.network_code = network_code.clone().detach()
        else:
            if self.network_code is None:
                network_code = self.forward_embedding().reshape(-1, self.level_dim)
                self.network_code = network_code.clone().detach()
            else:
                network_code = self.network_code

        outputs = hash_encode(inputs, network_code, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)        
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs






########### triplane decoder

#from models.embedder import get_embedder
class TriplaneEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=4, level_dim=4, per_level_scale=2, base_resolution=32, log2_hashmap_size=24, desired_resolution=None):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        assert per_level_scale == 2

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim * 3

        if level_dim % 2 != 0:
            print('[WARN] detected HashGrid level_dim % 2 != 0, which will cause very slow backward is also enabled fp16! (maybe fix later)')

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            print("=============triplane: level %d, resolution %d", resolution)
            params_in_level = min(self.max_params, (resolution) ** 2) # limit max number
            #params_in_level = np.ceil(params_in_level / 8) * 8 # make divisible
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        for plane in ["xy", "xz", "yz"]:
            embeddings = nn.Parameter(torch.empty(offset, level_dim))
            std = 1e-4
            embeddings.data.uniform_(-std, std)
            print(embeddings.shape)
            setattr(self, "embedding_%s"%(plane), embeddings)

        ######### triplane network

        self.out_feature_grid = [8, 8, 4] # h, w, c
        self.out_feature_dim = 8*8*4
        
        self.hidden_size = 512
        self.num_layers = 5

        multires = 6
        embed_fn, input_ch = get_embedder(multires, input_dims=2+1) # x, y scale
        self.embed_fn_fine = embed_fn


        grid_inputs = []
        grid_offsets = []
        grid_offset = 0
        for idx, scale in enumerate([4, 8, 16, 32]): 
            X = torch.linspace(-1, 1, scale*2+1)[1::2]
            Y = torch.linspace(-1, 1, scale*2+1)[1::2]

            xx, yy = torch.meshgrid(X, Y)

            inputs = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1)
            scale_input = torch.ones_like(inputs)[:, :1] * 2 * float(idx) / self.num_levels
            inputs = torch.cat((inputs, scale_input), dim=-1)
            inputs = self.embed_fn_fine(inputs)
            print(inputs.shape)
            grid_inputs.append(inputs)

            grid_offsets.append(grid_offset)
            grid_offset += inputs.shape[0]
        grid_offsets.append(grid_offset)
        grid_inputs = torch.cat(grid_inputs, dim=0)
        self.register_buffer('grid_inputs', grid_inputs)
        grid_offsets = torch.from_numpy(np.array(grid_offsets, dtype=np.int32))
        self.register_buffer('grid_offsets', grid_offsets)
        

        for plane in ["xy", "xz", "yz"]:
            input_ch = 16
            feature = torch.ones_like(self.grid_inputs[:,:input_ch])
            feature.uniform_(-1e-4, 1e-4)
            feature = torch.nn.Parameter(feature)
            setattr(self, "feature_%s"%(plane), feature)

            mlp = nn.Sequential(nn.Linear(input_ch, self.hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.hidden_size, self.hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.hidden_size, self.hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.hidden_size, self.hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.hidden_size, self.out_feature_dim)
                    )
            setattr(self, "mlp_%s"%(plane), mlp)

        self.cached = False


    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"HashEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} base_resolution={self.base_resolution} per_level_scale={self.per_level_scale} params={tuple(self.embeddings.shape)}"
    
    def forward_embedding(self):
        for plane in ["xy", "xz", "yz"]:
            mlp = getattr(self, "mlp_%s"%(plane))
            inputs = self.grid_inputs
            inputs = getattr(self, "feature_%s"%(plane))
            x = mlp(inputs)
            # reshape
            embeddings = []
            for level in range(self.grid_offsets.shape[0] - 1):
                cur_embedding = x[self.grid_offsets[level]:self.grid_offsets[level+1]]
                resolution = int(np.sqrt(cur_embedding.shape[0]))
                
                cur_embedding = cur_embedding.reshape(resolution, resolution, *self.out_feature_grid) # H, W, h, w, F
                #print(resolution, cur_embedding.shape)
                
                cur_embedding = cur_embedding.permute(0, 2, 1, 3, 4).contiguous().reshape(-1, self.out_feature_grid[-1])
                #print(cur_embedding.shape)
                embeddings.append(cur_embedding)
                

            embeddings = torch.cat(embeddings, dim=0)
            #print(embeddings.shape)
            setattr(self, "embedding_%s"%(plane), embeddings)
        return 

    def save_embedding_to_cache(self):
        for plane in ["xy", "xz", "yz"]:
            embeddings = getattr(self, "embedding_%s"%(plane))
            setattr(self, "cache_embedding_%s"%(plane), embeddings.clone().detach())
    
    def load_embedding_from_cache(self):
        for plane in ["xy", "xz", "yz"]:
            embeddings = getattr(self, "cache_embedding_%s"%(plane))
            setattr(self, "embedding_%s"%(plane), embeddings)
        
    def forward(self, inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + size) / (2 * size) # map to [0, 1]
        #print(inputs)        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        
        '''
        #TODO cache
        if inputs.requires_grad:
            self.forward_embedding()
            #self.save_embedding_to_cache()
            self.cached = False
            print("forward with inputs gradients=====")
        else:
            if not self.cached:
                self.forward_embedding()
                self.save_embedding_to_cache()
                self.cached = True
            else:
                self.load_embedding_from_cache()
        '''

        outputs = []
        for plane, index in zip(["xy", "xz", "yz"], [[0, 1], [0, 2], [1, 2]]):
            points_2d = inputs[:, index].contiguous()
            embedding = getattr(self, "embedding_%s"%(plane))
            output = hash_encode(points_2d, embedding, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
            #print(output.shape)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    def torch_forward(self,inputs, size=1):
        # inputs: [..., input_dim], normalized real world positions in [-size, size]
        # return: [..., num_levels * level_dim]
        #assert self.offsets.shape[0] == 2
        inputs = (inputs + size) / (2 * size) # map to [0, 1]
        #print(inputs)        
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())
        
        input_dim = inputs.shape[-1]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
      
        outputs = []
        for level in range(self.offsets.shape[0] - 1):
        
            S = np.log2(self.per_level_scale)
            scale = np.exp2(level * S) * self.base_resolution - 1
            resolution = ceil(scale) + 1
            #print("scale resolution")
            #print(scale, resolution, self.base_resolution)
            
            #import pdb; pdb.set_trace()
            
            # compute_index
            pos = inputs * scale #+ 0.5
            left = pos.long()
            wb = pos - left
            wb = (wb * wb) * (3. - 2. * wb)
            wa = 1.0 - wb
            
            #import pdb; pdb.set_trace()
            if input_dim == 3:
                volume = self.embeddings[self.offsets[level]:self.offsets[level+1]].reshape(resolution, resolution, resolution, -1).permute(2, 1, 0, 3).contiguous()

                #print(wa)
                #print(wb)
                lx, ly, lz = left.unbind(-1)

                v000 = volume[lx, ly, lz]
                v001 = volume[lx, ly, lz + 1]
                v010 = volume[lx, ly + 1, lz]
                v011 = volume[lx, ly + 1, lz + 1]
                v100 = volume[lx + 1, ly, lz]
                v101 = volume[lx + 1, ly, lz + 1]
                v110 = volume[lx + 1, ly + 1, lz]
                v111 = volume[lx + 1, ly + 1, lz + 1]

                c00 = v000 * wa[:, 2:] + v001 * wb[:, 2:]
                c01 = v010 * wa[:, 2:] + v011 * wb[:, 2:]
                c10 = v100 * wa[:, 2:] + v101 * wb[:, 2:]
                c11 = v110 * wa[:, 2:] + v111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                output = c0 * wa[:, :1] + c1 * wb[:, :1]
                outputs.append(output)
            elif input_dim == 2:
                volume = self.embeddings[self.offsets[level]:self.offsets[level+1]].reshape(resolution, resolution, -1).permute(1, 0, 2).contiguous()

                #print(wa)
                #print(wb)
                lx, ly = left.unbind(-1)
                print(lx.min(),lx.max())
                c00 = volume[lx, ly]
                c01 = volume[lx, ly + 1]
                c10 = volume[lx + 1, ly]
                c11 = volume[lx + 1, ly + 1]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                output = c0 * wa[:, :1] + c1 * wb[:, :1]
                outputs.append(output)
            else:
                raise NotImplementedError

        outputs = torch.cat((outputs), dim=-1)
        #import pdb; pdb.set_trace()
        
        #outputs = hash_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs
