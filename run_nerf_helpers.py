import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import math

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        min_freq = self.kwargs['min_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        open_res = self.kwargs['open_res']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(min_freq, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**min_freq, 2.**max_freq, steps=N_freqs)

        effective_freq = 0
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class MipEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x[:,:d])
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        min_freq = self.kwargs['min_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands_y = 2.**torch.linspace(min_freq, max_freq, steps=N_freqs)
            freq_bands_w = 4.**torch.linspace(min_freq, max_freq, steps=N_freqs)
        else:
            freq_bands_y = torch.linspace(2.**min_freq, 2.**max_freq, steps=N_freqs)
            freq_bands_w = torch.linspace(4.**min_freq, 4.**max_freq, steps=N_freqs)

        for ctr in range(len(freq_bands_y)):
            for p_fn in self.kwargs['periodic_fns']: 
                embed_fns.append(lambda inputs, p_fn=p_fn, freq_y=freq_bands_y[ctr], freq_w=freq_bands_w[ctr] : p_fn(inputs[:,:d] * freq_y) * torch.exp((-0.5) * freq_w * inputs[:,d:]))
                out_dim += d
                
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

import math
def integrated_pos_enc(x_coord, min_freq, max_freq, N_freqs):
    x, x_cov_diag = x_coord[:,:3], x_coord[:,3:]
    scales = 2.**torch.linspace(min_freq, max_freq, steps=N_freqs)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)

    embedding = expected_sin(
        torch.cat([y, y + 0.5 * math.pi], axis=-1),
        torch.cat([y_var] * 2, axis=-1))[0]

    return embedding


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = torch.maximum(
        torch.tensor(0), 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2)
    return y, y_var


def get_embedder(multires, open_res, min_multires=0, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'min_freq_log2': min_multires,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
                'open_res' : open_res,
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def get_mip_embedder(multires, min_multires=0, i=0, include_input=True, log_sampling=True):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : include_input,
                'input_dims' : 3,
                'min_freq_log2': min_multires,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : log_sampling,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = MipEmbedder(**embed_kwargs)
    embed = lambda inputs, eo=embedder_obj : eo.embed(inputs)
    return embed, embedder_obj.out_dim

class Bungee_NeRF_baseblock(nn.Module):
    def __init__(self, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_baseblock, self).__init__()
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, net_width)] + [nn.Linear(net_width, net_width) for _ in range(3)])
        self.views_linear = nn.Linear(input_ch_views + net_width, net_width//2)
        self.feature_linear = nn.Linear(net_width, net_width)
        self.alpha_linear = nn.Linear(net_width, 1)
        self.rgb_linear = nn.Linear(net_width//2, 3)

    def forward(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = torch.cat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = F.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h


class Bungee_NeRF_resblock(nn.Module):
    def __init__(self, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_resblock, self).__init__()
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch+net_width, net_width), nn.Linear(net_width, net_width)])
        self.views_linear = nn.Linear(input_ch_views + net_width, net_width//2)
        self.feature_linear = nn.Linear(net_width, net_width)
        self.alpha_linear = nn.Linear(net_width, 1)
        self.rgb_linear = nn.Linear(net_width//2, 3)
    
    def forward(self, input_pts, input_views, h):
        h = torch.cat([input_pts, h], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = torch.cat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = F.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h


class Bungee_NeRF_block(nn.Module):
    def __init__(self, num_resblocks=3, net_width=256, input_ch=3, input_ch_views=3):
        super(Bungee_NeRF_block, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.num_resblocks = num_resblocks

        self.baseblock = Bungee_NeRF_baseblock(net_width=net_width, input_ch=input_ch, input_ch_views=input_ch_views)
        self.resblocks = nn.ModuleList([
            Bungee_NeRF_resblock(net_width=net_width, input_ch=input_ch, input_ch_views=input_ch_views) for _ in range(num_resblocks)
        ])

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        alphas = []
        rgbs = []
        base_rgb, base_alpha, h = self.baseblock(input_pts, input_views)
        alphas.append(base_alpha)
        rgbs.append(base_rgb)
        for i in range(self.num_resblocks):
            res_rgb, res_alpha, h = self.resblocks[i](input_pts, input_views, h)
            alphas.append(res_alpha)
            rgbs.append(res_rgb)

        output = torch.cat([torch.stack(rgbs,1),torch.stack(alphas,1)],-1)
        return output

def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    dirs = dirs/torch.norm(dirs, dim=-1)[...,None]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    dirs = dirs/np.linalg.norm(dirs, axis=-1)[..., None]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def get_radii_for_test(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[np.newaxis, ..., np.newaxis, :] * c2w[:, np.newaxis, np.newaxis, :3,:3], -1) 
    dx = torch.sqrt(
        torch.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :])**2, -1))
    dx = torch.cat([dx, dx[:, -2:-1, :]], 1)
    radii = dx[..., None] * 2 / np.sqrt(12)
    return radii

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    eps = 1e-5
    weight_sum = torch.sum(weights, axis=-1, keepdims=True)
    padding = torch.maximum(torch.tensor(0), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = torch.minimum(torch.tensor(1), torch.cumsum(pdf[..., :-1], axis=-1))

    cdf = torch.cat([
            torch.zeros(list(cdf.shape[:-1]) + [1]), cdf,
            torch.ones(list(cdf.shape[:-1]) + [1])
    ], axis=-1)

    if randomized:
        s = 1 / num_samples
        u = np.arange(num_samples) * s
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
        jitter = np.random.uniform(high=s - np.finfo('float32').eps, size=list(cdf.shape[:-1]) + [num_samples])
        u = u + jitter
        u = np.minimum(u, 1. - np.finfo('float32').eps)
    else:
        u = np.linspace(0., 1. - np.finfo('float32').eps, num_samples)
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    u = torch.from_numpy(u).to(cdf)
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), dim=-2)[0]
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), dim=-2)[0]
        return x0, x1


    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = (u - cdf_g0) / (cdf_g1 - cdf_g0)
    t[t != t] = 0
    t = torch.clamp(t, 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples



