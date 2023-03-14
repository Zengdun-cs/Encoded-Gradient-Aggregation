import torch
from utils import tensor_split

class GradEncoder():
    def __init__(self, encoder, code_block) -> None:
        self.encoder = encoder
        self.encoder.eval()
        self.code_block = code_block  # encoder输入维度

    def __call__(self, gradient, norm, interval):
        clipped_grad = torch.clamp(gradient, 0, norm)
        quantized_grad = self._quantize(gradient, norm, interval)
        encoded_grad = self._encode_gradient(quantized_grad)
        return quantized_grad, encoded_grad

    def _quantize(self, tensor, norm, interval):
        sign = torch.sign(tensor)
        
        normalized_ = torch.abs(tensor) / norm
        scaled_ = normalized_ * interval
        
        # random
        l = torch.clamp(scaled_, 0, interval - 1).ceil()
        probabilities = scaled_ - l.type(torch.float32)
        r = torch.rand(l.size())
        l[:] += (probabilities > r).type(torch.float32)

        return l*sign

    def _encode_gradient(self, quantized_grad):
        to_encode = tensor_split(quantized_grad, self.code_block)
        encoded_grad = self.encoder(to_encode).view(-1)
        return encoded_grad


class GradDecoder():
    def __init__(self, decoder, code_block) -> None:
        self.decoder = decoder
        self.decoder.eval()
        self.code_block = code_block  # encoder输入维度

    def __call__(self, code, norm, interval):
        quantized_grad = self._decode_gradient(code)

        # random
        sign = torch.sign(quantized_grad)
        quantized_grad = quantized_grad.abs()

        l = torch.clamp(quantized_grad, 0, interval - 1).type(torch.int32)
        probabilities = quantized_grad - l.type(torch.float32)
        r = torch.rand(l.size())
        l[:] += (probabilities > r).type(torch.int32)

        gradient = self._dequantize(l, norm, interval).view(-1) * sign.type(torch.float32)
        return l*sign, gradient

    def _decode_gradient(self, encoded_gradient):
        to_decode = tensor_split(encoded_gradient, self.code_block)
        decoded_ = self.decoder(to_decode).view(-1)
        return decoded_

    def _dequantize(self, tensor, norm, interval):
        tensor = torch.clamp(tensor, 0, interval).to(torch.float32)
        dequantize_ = tensor * norm / interval
        return dequantize_

def list_split(tensor, n, fill_last=True):
    tensor_list = [tensor[i:i + n] for i in range(0, tensor.shape[0], n)]

    if fill_last is True and tensor_list[-1].shape[0] != n:
        len = tensor_list[-1].shape[0]
        zeros = torch.zeros(size=(n - len, ))
        tensor_list[-1] = torch.cat([tensor_list[-1], zeros])

    return tensor_list


class BlockWiseEncoder(GradEncoder):
    def __init__(self, encoder, code_block) -> None:
        super().__init__(encoder, code_block)

    def __call__(self, gradient, interval):
        print("ZERO rate {}/{}".format(torch.nonzero(gradient).numel(), gradient.numel()))
        blocks = list_split(gradient, self.code_block)

        norms = [max(block.max(), 1) for block in blocks]
        quantized_ = [
            self._quantize(block, norm, interval)
            for norm, block in zip(norms, blocks)
        ]

        encoded = self.encoder(torch.stack(quantized_)).view(-1)

        return encoded, norms, torch.stack(quantized_).view(-1)

    def _quantize(self, tensor, norm, interval):
        return super()._quantize(tensor, norm, interval)


class BlockWiseDecoder(GradDecoder):
    def __init__(self, decoder, code_block) -> None:
        super().__init__(decoder, code_block)

    def __call__(self, code, norms, interval):
        blocks = list_split(code, self.code_block)
        decoded = self.decoder(torch.stack(blocks))
        dequant = torch.cat([
            self._dequantize(quant, norm, interval)
            for quant, norm in zip(decoded, norms)
        ])
        return dequant

    def _dequantize(self, tensor, norm, interval):
        return super()._dequantize(tensor, norm, interval)


class PipelineAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, interval):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.interval = interval

    def forward(self, data):
        code = self.encoder(data)
        combined_code = torch.mean(code, dim=1)
        decode = self.decoder(combined_code) * self.interval
        return decode