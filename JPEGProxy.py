# Implement JPEG using Pytorch as a proxy for JPEG compression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def dct_mat_coeff(L=8):
    C = np.zeros((L,L))
    for k in range(L):
        for n in range(L):
            if k == 0:
                C[k,n] = np.sqrt(1/L)
            else:
                C[k,n] = np.sqrt(2/L)*np.cos((np.pi*k*(1/2+n))/L)
    return C

class DCTLayer(nn.Module):
    def __init__(self, cdim, rdim):
        super(DCTLayer, self).__init__()
        self.register_buffer('dct_mat_col', torch.from_numpy(dct_mat_coeff(cdim)).float())
        self.register_buffer('dct_mat_row', torch.from_numpy(dct_mat_coeff(rdim)).float())
    
    def forward(self, x):
        assert len(x.shape) == 3
        # x.shape would be N, 8, 8
        out = torch.einsum('ij,bjk->bik', self.dct_mat_col, x)
        out = out.transpose(2,1)
        out = torch.einsum('ij,bjk->bik', self.dct_mat_row, out)
        out = out.transpose(2,1)
        return out

class iDCTLayer(nn.Module):
    def __init__(self, cdim, rdim):
        super(iDCTLayer, self).__init__()
        self.register_buffer('dct_mat_col', torch.from_numpy(dct_mat_coeff(cdim).T).float())
        self.register_buffer('dct_mat_row', torch.from_numpy(dct_mat_coeff(rdim).T).float())
    
    def forward(self, x):
        assert len(x.shape) == 3
        # x.shape would be N, 8, 8
        out = x.transpose(2,1)
        out = torch.einsum('ij,bjk->bik', self.dct_mat_row, out)
        out = out.transpose(2,1)
        out = torch.einsum('ij,bjk->bik', self.dct_mat_col, out)
        return out


class BypassRound(Function):
  @staticmethod
  def forward(ctx, inputs):
    return torch.round(inputs)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

bypass_round = BypassRound.apply

class BlockDCTModule(nn.Module):
    """Apply block-wise DCT over torch images.
    """
    def __init__(self, block_size=(8, 8)) -> None:
        super().__init__()
        self.DCT = DCTLayer(*block_size)
        self.block_size = block_size
    
    def forward(self, x):
        b, c, h, w = x.shape
        blk_h, blk_w = self.block_size
        assert h % self.block_size[0] == 0
        assert w % self.block_size[1] == 0
        blocks = F.unfold(x,
                          kernel_size=self.block_size,
                          stride=self.block_size)
        blocks = blocks.permute(0, 2, 1).reshape(
            -1, c, self.block_size[0], self.block_size[1])
        channels = blocks.split(1, dim=1)
        assert len(channels) == c
        coeff_channels = [
            self.DCT(channel[:, 0]) for channel in channels
        ]
        coeffs = torch.stack(coeff_channels, dim=3) # N, 8, 8, c
        # Organize coeffs in (B, h//8, w//8, 8, 8, c)
        n_blocks_col = h // self.block_size[0]
        n_blocks_row = w // self.block_size[1]
        coeffs = coeffs.reshape(
            -1, n_blocks_col, n_blocks_row, blk_h * blk_w * c)
        assert coeffs.shape[0] == b
        coeffs = coeffs.permute(0, 3, 1, 2)
        return coeffs

class BlockIDCTModule(nn.Module):
    """Apply block-wise IDCT over torch images.
    """
    def __init__(self, block_size=(8, 8)) -> None:
        super().__init__()
        self.IDCT = iDCTLayer(*block_size)
        self.block_size = block_size
    
    def forward(self, x):
        b, chw, n_bloclk_h, n_block_w = x.shape
        c = chw // self.block_size[0] // self.block_size[1]
        blk_h, blk_w = self.block_size
        orig_h, orig_w = n_bloclk_h * blk_h, n_block_w * blk_w
        coeffs = x.permute(0, 2, 3, 1).reshape(
            -1, self.block_size[0], self.block_size[1], c)
        channels = coeffs.split(1, dim=3)
        assert len(channels) == c
        blocks_channels = [
            self.IDCT(channel[..., 0]) for channel in channels
        ]
        blocks = torch.stack(blocks_channels, dim=3) # N, 8, 8, c
        n_blocks_per_frame = n_bloclk_h * n_block_w
        c = chw // blk_h // blk_w
        blocks = blocks.reshape(-1, n_blocks_per_frame, blk_h, blk_w, c)
        blocks = blocks.permute(0, 4, 2, 3, 1).contiguous().reshape(b, c * blk_h * blk_w, n_blocks_per_frame)
        x_recon = F.fold(blocks,
                         output_size=(orig_h, orig_w),
                         kernel_size=self.block_size,
                         stride=self.block_size)
        return x_recon


def bicubic_upsampler(img, scale_factor=2):
    img_up = F.interpolate(img, scale_factor=scale_factor, mode='bicubic', align_corners=False)
    return img_up

BT601_RGB2YUV_MAT = np.array(
    [[0.299, 0.587, 0.114],
     [-0.168736, -0.331264, 0.5],
     [0.5, -0.418688, -0.081312]])


BT601_YUV2RGB_MAT = np.array(
    [[1.0, 0.0, 1.402],
     [1.0, -0.344136, -0.714136],
     [1.0, 1.772, 0.0]])

class RGB2YUV(nn.Module):
    """RGB to YUV conversion module.
    
    Assume input is torch Tensor in range [0, 255]."""
    def __init__(self,bit_depth=8):
        super(RGB2YUV, self).__init__()
        self.bit_depth = bit_depth
        rgb2yuv_mat = torch.from_numpy(BT601_RGB2YUV_MAT).float()
        if self.bit_depth == 10:
            uv_offset = torch.tensor([0, 512, 512]).float()
        else:
            uv_offset = torch.tensor([0, 128, 128]).float()
        self.register_buffer('rgb2yuv_mat', rgb2yuv_mat)
        self.register_buffer('uv_offset', uv_offset)
    
    def forward(self, x):
        '''Convert RGB image to YUV image.'''
        yuv = torch.einsum('bchw,cd->bdhw', x, self.rgb2yuv_mat.t())
        return yuv + self.uv_offset.view(1, 3, 1, 1)
    
class YUV2RGB(nn.Module):
    """YUV to RGB conversion module.
    
    Assume input is torch Tensor in range [0, 255]."""
    def __init__(self):
        super(YUV2RGB, self).__init__()
        yuv2rgb_mat = torch.from_numpy(BT601_YUV2RGB_MAT).float()
        uv_offset = torch.tensor([0, 128, 128]).float()
        self.register_buffer('yuv2rgb_mat', yuv2rgb_mat)
        self.register_buffer('uv_offset', uv_offset)
    
    def forward(self, x):
        '''Convert YUV image to RGB image.'''
        rgb = torch.einsum('bchw,cd->bdhw', x - self.uv_offset.view(1, 3, 1, 1), self.yuv2rgb_mat.t())
        return rgb

def soft_L1(x: torch.Tensor):
    '''Soft L1 function.'''
    return torch.sqrt(torch.pow(x, 2) + 1e-6)

class JPEGProxy(nn.Module):
    """JPEG proxy module.

    This module is used to implement JPEG compression as a proxy for JPEG compression. Currently we use a constant quantization table rather than the
    one used in JPEG standard.
    """
    def __init__(self,
                 block_size=(8, 8),
                 is_rgb=False,
                 is_yuv420=False,
                 debug_mode=False,
                 use_noisy_quant=False,
                 clip_to_255=False,
                 clip_to_1023=False,
                 per_channel_jpeg=True,
                 use_qtable=True):
        super(JPEGProxy, self).__init__()
        self.use_qtable = use_qtable
        assert (block_size == (8, 8) and use_qtable) or not use_qtable, 'Only support 8x8 block size with qtable'
        self.block_size = block_size
        self.dct_module_4 = BlockDCTModule((4, 4))
        self.idct_module_4 = BlockIDCTModule((4, 4))
        self.dct_module_8 = BlockDCTModule((8, 8))
        self.idct_module_8 = BlockIDCTModule((8, 8))
        self.dct_module_16 = BlockDCTModule((16, 16))
        self.idct_module_16 = BlockIDCTModule((16, 16))
        self.dct_module_32 = BlockDCTModule((32, 32))
        self.idct_module_32 = BlockIDCTModule((32, 32))
        self.dct_module_dict = {
            4: self.dct_module_4,
            8: self.dct_module_8,
            16: self.dct_module_16,
            32: self.dct_module_32
        }
        self.idct_module_dict = {
            4: self.idct_module_4,
            8: self.idct_module_8,
            16: self.idct_module_16,
            32: self.idct_module_32
        }
        self.is_rgb = is_rgb
        self.is_yuv420 = is_yuv420
        if is_rgb and is_yuv420:
            raise ValueError('Cannot be both RGB and YUV420')
        self.rgb2yuv = RGB2YUV()
        self.yuv2rgb = YUV2RGB()
        self.debug_mode = debug_mode
        self.use_noisy_quant = use_noisy_quant
        self.clip_to_255 = clip_to_255
        self.clip_to_1023 = clip_to_1023
        self.per_channel_jpeg = per_channel_jpeg
        if self.clip_to_255 and clip_to_1023:
            raise ValueError('Cannot be both 8bit and 10bit')

        IJG_table = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)
        
        # register IJG table as buffer
        self.register_buffer('IJG_table', torch.from_numpy(IJG_table).float())

    def get_jpeg_quantization_table(self, qf):
        if qf < 50:
            scale = 5000 / qf
        else:
            scale = 200 - 2 * qf
        table = (scale * self.IJG_table + 50) / 100
        return table

    def run_tf_jpeg_encode(self, x: np.array, qf=100):
        x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        bit_stream = tf.io.encode_jpeg(
            tf.convert_to_tensor(x),
            quality=qf,
            progressive=False,
            optimize_size=False,
            chroma_downsampling=False
        ).numpy()
        return bit_stream

    def get_rate_with_real_jpeg(self, x, qf=None, qstep=1):
        '''Get bit-rate with real JPEG compression.'''
        if qf is None:
            qf = 100 - (100 * qstep -  50) / 32
        with torch.no_grad():
            if x.shape[1] == 1:
                bitstreams = [
                    self.run_tf_jpeg_encode(x[i].permute(1, 2, 0).cpu().numpy(), qf=qf)
                    for i in range(x.shape[0])
                ]
                bitrates = [len(b) * 8 for b in bitstreams]
                bitrates = torch.tensor(bitrates).to(x.device)
                return bitrates
            else:
                if self.per_channel_jpeg:
                    tf_x = x.permute(0, 2, 3, 1).cpu().numpy()
                    bit_streams = []
                    for i in range(tf_x.shape[0]):
                        chan_bitstreams = [
                            self.run_tf_jpeg_encode(tf_x[i, ..., j:j+1], qf=qf)
                            for j in range(tf_x.shape[-1])
                        ]
                        bitstream = b''.join(chan_bitstreams)
                        bit_streams.append(bitstream)
                    bitrates = [len(b) * 8 for b in bit_streams]
                    bitrates = torch.tensor(bitrates).to(x.device)
                    return bitrates
                else:
                    assert x.shape[1] == 3, 'Only support RGB or YUV420'
                    rgb_x = self.yuv2rgb(x)
                    rgb_x = torch.clamp(torch.round(rgb_x), 0, 255)
                    bitstreams = [
                        self.run_tf_jpeg_encode(rgb_x[i].permute(1, 2, 0).cpu().numpy(), qf=qf)
                        for i in range(rgb_x.shape[0])
                    ]
                    bitrates = [len(b) * 8 for b in bitstreams]
                    bitrates = torch.tensor(bitrates).to(x.device)
                    return bitrates
    
    def get_flatten_quant_table(self, qf, dim=1):
        '''Get flatten quantization table.'''
        table = self.get_jpeg_quantization_table(qf)
        if self.block_size == (8, 8):
            table = table.view(1, 1, 8, 8)
        elif self.block_size == (16, 16):
            table = table.view(1, 1, 16, 16)
        table = torch.cat(
            [table,] * dim, dim=1).reshape(
            1, dim * self.block_size[0] * self.block_size[1], 1, 1)
        return table
    
    def quantize(self, dct_coeff, qstep):
        if self.use_noisy_quant:
            noise = torch.rand_like(dct_coeff) - 0.5
            quant_coeff = dct_coeff / qstep + noise
        else:
            quant_coeff = bypass_round(dct_coeff / qstep)
        dequant_coeff = quant_coeff * qstep
        return dequant_coeff

    def quantize_with_q_table(self, dct_coeff, qf):
        dim = dct_coeff.shape[1] // self.block_size[0] // self.block_size[1]
        qtable = self.get_flatten_quant_table(qf, dim=dim)
        if self.use_noisy_quant:
            noise = torch.rand_like(dct_coeff) - 0.5
            quant_coeff = dct_coeff / qtable + noise
        else:
            quant_coeff = bypass_round(dct_coeff / qtable)
        dequant_coeff = quant_coeff * qtable
        return dequant_coeff
    
    def get_dc_qstep(self, qf):
        if qf < 50:
            scale = 5000 / qf
        else:
            scale = 200 - 2 * qf
        # scale = 5000 / qf
        qstep = (scale * 16 + 50) / 100
        return qstep

    def forward(self, x, qstep=None, qf=None, uv_qstep=None, blk_size=8):
        '''Forward pass: calculate reconstructed x and estimated bit-rate.'''
        # Pad x to be multiple of block size if necessary
        pad_h = (
            self.block_size[0] - x.shape[2] % self.block_size[0]
            ) % self.block_size[0]
        pad_w = (
            self.block_size[1] - x.shape[3] % self.block_size[1]
            ) % self.block_size[1]
        padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        if self.clip_to_255:
            padded_x = torch.clamp(padded_x, 0, 255)
        if self.clip_to_1023:
            padded_x = torch.clamp(padded_x, 0, 1023)

        if not self.is_yuv420:
            if self.is_rgb:
                padded_x = self.rgb2yuv(padded_x)
            dct_coeff = self.dct_module_dict[blk_size](padded_x)
            if qf is None:
                dequant_coeff = self.quantize(dct_coeff, qstep)
            else:
                assert self.use_qtable, 'Can only use qtable with qf'
                dequant_coeff = self.quantize_with_q_table(dct_coeff, qf)
            recon_x = self.idct_module_dict[blk_size](dequant_coeff)
            if self.is_rgb:
                recon_x = self.yuv2rgb(recon_x)
            # Estimate bit-rate with entropy model
            coeff_l1 = soft_L1(dct_coeff)
            if qf is None:
                gross_rate = torch.log2(1 + coeff_l1 / qstep).sum(dim=(1, 2, 3))
            else:
                gross_rate = torch.log2(1 + coeff_l1 / self.get_dc_qstep(qf)).sum(dim=(1, 2, 3))
        else:
            if uv_qstep is None:
                uv_qstep = qstep
            padded_x_y = padded_x[:, 0:1, :, :]
            padded_x_uv = bicubic_upsampler(
                padded_x[:, 1:, :, :], scale_factor=0.5)
            y_dct_ceoff = self.dct_module_dict[blk_size](padded_x_y)
            if qf is None:
                dequant_y_coeff = self.quantize(y_dct_ceoff, qstep)
            else:
                assert self.use_qtable, 'Can only use qtable with qf'
                dequant_y_coeff = self.quantize_with_q_table(y_dct_ceoff, qf)
            recon_y = self.idct_module_dict[blk_size](dequant_y_coeff)
            
            uv_dct_coeff = self.dct_module_dict[blk_size](padded_x_uv)
            if qf is None:
                dequant_uv_coeff = self.quantize(uv_dct_coeff, uv_qstep)
            else:
                assert self.use_qtable, 'Can only use qtable with qf'
                dequant_uv_coeff = self.quantize_with_q_table(uv_dct_coeff, qf)
            recon_uv = self.idct_module_dict[blk_size](dequant_uv_coeff)
            recon_uv = bicubic_upsampler(recon_uv, scale_factor=2)
            recon_x = torch.cat([recon_y, recon_uv], dim=1)
            # Estimate bit-rate with entropy model
            y_coeff_l1 = soft_L1(y_dct_ceoff)
            uv_coeff_l1 = soft_L1(uv_dct_coeff)
            gross_rate = (
                torch.log2(1 + y_coeff_l1 / qstep).sum(dim=(1, 2, 3)) +
                torch.log2(1 + uv_coeff_l1 / uv_qstep).sum(dim=(1, 2, 3))
                )

        # Calculate real bit-rate
        real_bitrate = self.get_rate_with_real_jpeg(padded_x//4, qf=qf, qstep=qstep)
        a = (real_bitrate / gross_rate).detach()
        # Calculate estimated bit-rate
        est_bitrate = gross_rate * a
        recon_x = recon_x[:, :, :x.shape[2], :x.shape[3]]
        
        if self.debug_mode:
            a_list = a
            return recon_x, est_bitrate, a_list, real_bitrate, dct_coeff, dequant_coeff

        if self.clip_to_255:
            recon_x = torch.clamp(recon_x, 0, 255)
        return recon_x, est_bitrate

class JPEG_Module(nn.Module):
    def __init__(self, is_yuv=True) -> None:
        super().__init__()
        self.rgb2yuv = RGB2YUV()
        self.yuv2rgb = YUV2RGB()
        self.is_yuv = is_yuv
    
    def jpeg_444_compress(self, x: torch.Tensor, qf: int):
        with torch.no_grad():
            x_yuv = x.cpu().numpy().transpose(0, 2, 3, 1)
            x_yuv = np.clip(np.round(x_yuv), 0, 255).astype(np.uint8)
            num_bits = []
            recons = []
            for i in range(x_yuv.shape[0]):
                chan_bit_streams = [
                    tf.io.encode_jpeg(
                        tf.convert_to_tensor(x_yuv[i, ..., j:j+1]),
                            quality=qf,
                            progressive=False,
                            optimize_size=True,
                            chroma_downsampling=False
                        ).numpy() for j in range(x_yuv.shape[-1])
                ]
                bit_stream = b''.join(chan_bit_streams)
                img_bits = len(bit_stream) * 8
                img_recon = np.concatenate([tf.io.decode_jpeg(bin).numpy() for bin in chan_bit_streams], -1)
                img_recon = torch.from_numpy(img_recon).permute(2, 0, 1).float()
                recons.append(img_recon)
                num_bits.append(np.sum(img_bits))
            recons = torch.stack(recons, dim=0).to(x.device)
            num_bits = torch.from_numpy(
                np.array(num_bits, dtype=np.float32)).to(x.device)
            return recons, num_bits

    def forward(self, x, qf):
        return self.jpeg_444_compress(x, qf)

