from typing import Any, Optional, Union, List
from torch import nn, Tensor
from torch.ao.nn import quantized as nnq
from torch.ao.quantization import fuse_modules, fuse_modules_qat, \
    QuantStub, DeQuantStub
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


def _fuse_modules(model: nn.Module, modules_to_fuse: Union[List[str], List[List[str]]], 
    is_qat: Optional[bool], **kwargs: Any):
    if is_qat is None:
        is_qat = model.training
    fuse_method = fuse_modules_qat if is_qat else fuse_modules
    return fuse_method(model, modules_to_fuse, **kwargs)

class QBottleneck(Bottleneck):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add_relu = nnq.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        indentity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out
    
    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(self,
            [["conv1", "bn1", "relu1"],
             ["conv2", "bn2", "relu2"],
             ["conv3", "bn3"]],
            is_qat,
            inplace=True
        )
        if self.downsample:
            _fuse_modules(self.downsample,
                ["0", "1"],
                is_qat,
                inplace=True
            )

    class QResNet(ResNet):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        def forward(self, x: Tensor) -> Tensor:
            x = self.quant(x)
            x = self._forward_impl(x)
            x = self.dequant(x)
            return x

        def fuse_model(self, is_qat: Optional[bool] = None) -> None:
            """
            Fuse conv+bn+relu / conv+bn / conv+relu modules
            to prepare for quantization
            """ 
            _fuse_modules(self,
                ["conv1", "bn1", "relu"],
                is_qat,
                inplace = True
            )
            for module in self.modules():
                if type(module)==QBottleneck:
                    module.fuse_model()