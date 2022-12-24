# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os

import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common import initializer as weight_init

from src.models.layers.drop_path import DropPath2D
from src.models.layers.identity import Identity

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


class Residual(nn.Cell):
    def __init__(self, fn, drop_path_rate):
        super().__init__()
        self.fn = fn
        self.drop_path_rate = drop_path_rate
        self.drop = DropPath2D(drop_prob=drop_path_rate)

    def construct(self, x):
        return self.drop(self.fn(x)) + x


class ConvMixer(nn.Cell):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, activation=nn.GELU,
                 drop_path_rate=0., **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.head = nn.Dense(in_channels=dim, out_channels=num_classes) if num_classes > 0 else Identity()
        self.stem = nn.SequentialCell([
            nn.Conv2d(in_channels=in_chans, out_channels=dim, kernel_size=patch_size, stride=patch_size,
                      pad_mode='same', has_bias=True),
            activation(),
            BatchNorm2d(num_features=dim, momentum=0.9)
        ])
        drop_path_rates = [i for i in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.SequentialCell(
            [nn.SequentialCell([
                Residual(nn.SequentialCell([
                    nn.Conv2d(dim, dim, kernel_size, group=dim, pad_mode="same", has_bias=True),
                    activation(),
                    BatchNorm2d(dim)
                ]), drop_path_rates[i]),
                nn.Conv2d(dim, dim, kernel_size=1, has_bias=True),
                activation(),
                BatchNorm2d(dim)
            ]) for i in range(depth)]
        )
        self.init_weights()

    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = ops.ReduceMean()(x, (2, 3))
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def convmixer_1536_20(**kwargs):
    return ConvMixer(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)


def convmixer_768_32(**kwargs):
    return ConvMixer(dim=768, depth=32, kernel_size=7, patch_size=7, activation=nn.ReLU, **kwargs)


def convmixer_1024_20_ks9_p14(**kwargs):
    return ConvMixer(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)


if __name__ == "__main__":
    from mindspore import context, Tensor
    from mindspore import dtype as mstype

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([2, 3, 224, 224]), dtype=mstype.float32)
    model = convmixer_1536_20()
    # out = model(data)
    # print(out.shape)
    params_num = 0
    for name, param in model.parameters_and_names():
        if "moving" in name:
            print(name, param.shape)
    for name, param in model.parameters_and_names():
        if "moving" not in name:
            print(name, param.shape)
    for name, param in model.parameters_and_names():
        params_num = np.prod(param.shape) + params_num
    print(params_num)
