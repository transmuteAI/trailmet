# MIT License
#
# Copyright (c) 2023 Transmute AI Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from .resnet import get_resnet_model
from .mobilenetv2_bireal import get_mobilenet as get_mobilenet_bireal
from .mobilenet import get_mobilenet as get_mobilenet_normal
from .resnet_bireal import make_birealnet18, make_birealnet34, make_birealnet50
from .resnet_chip import resnet_50 as resnet_50_chip

# All models_bnnbn
# reactnet-18
from .models_bnnbn.Qa_reactnet_18_bn import birealnet18 as Qa_reactnet_18_bn
from .models_bnnbn.Qa_reactnet_18_none import birealnet18 as Qa_reactnet_18_none
from .models_bnnbn.Qa_reactnet_18_bf import birealnet18 as Qa_reactnet_18_bf

from .models_bnnbn.Qaw_reactnet_18_bn import birealnet18 as Qaw_reactnet_18_bn
from .models_bnnbn.Qaw_reactnet_18_none import birealnet18 as Qaw_reactnet_18_none
from .models_bnnbn.Qaw_reactnet_18_bf import birealnet18 as Qaw_reactnet_18_bf

# reactnet-A
from .models_bnnbn.Qa_reactnet_A_bn import reactnet as Qa_reactnet_A_bn
from .models_bnnbn.Qa_reactnet_A_none import reactnet as Qa_reactnet_A_none
from .models_bnnbn.Qa_reactnet_A_bf import reactnet as Qa_reactnet_A_bf

from .models_bnnbn.Qaw_reactnet_A_bn import reactnet as Qaw_reactnet_A_bn
from .models_bnnbn.Qaw_reactnet_A_none import reactnet as Qaw_reactnet_A_none
from .models_bnnbn.Qaw_reactnet_A_bf import reactnet as Qaw_reactnet_A_bf


class ModelsFactory(object):

    @staticmethod
    def create_model(name,
                     num_classes=100,
                     pretrained=False,
                     version='original',
                     **kwargs):
        """Returns the requested model, ready for training/pruning with the
        specified method.

        Args:
            name: model name 'resnet18','resnet50'
            num_classes: number of classes
        Return:
            model object
        """

        if name in [
                'resnet18',
                'resnet20',
                'resnet32',
                'resnet50',
                'resnet56',
                'resnet101',
                'resnet110',
        ]:
            if version == 'original':
                assert 'insize' in kwargs, 'should provide input size'
                insize = kwargs['insize']
                model = get_resnet_model(name,
                                         num_classes,
                                         insize=insize,
                                         pretrained=pretrained)

            elif version == 'chip':
                assert 'sparsity' in kwargs, 'should provide sparsity for chip'
                model = resnet_50_chip(sparsity=eval(kwargs['sparsity']),
                                       num_classes=num_classes)
            elif version == 'bireal':
                assert 'insize' in kwargs, 'should provide input size'
                insize = kwargs['insize']
                assert 'num_fp' in kwargs, 'should provide num_fp'
                num_fp = kwargs['num_fp']
                if name == 'resnet18':
                    model = make_birealnet18(num_classes=num_classes,
                                             insize=insize,
                                             num_fp=num_fp)
                elif name == 'resnet34':
                    model = make_birealnet34(num_classes=num_classes,
                                             insize=insize,
                                             num_fp=num_fp)
                elif name == 'resnet50':
                    model = make_birealnet50(num_classes=num_classes,
                                             insize=insize,
                                             num_fp=num_fp)
                else:
                    raise Exception(
                        'unknown model {} for BirealNet, available .models_bnnbn are (resnet18, resnet34, resnet50)'
                        .format(name))

        elif name in ['mobilenetv2']:
            if version == 'original':
                model = get_mobilenet_normal(name, num_classes, **kwargs)
            elif version == 'bireal':
                assert 'num_fp' in kwargs, 'should provide num_fp'
                num_fp = kwargs['num_fp']
                model = get_mobilenet_bireal(num_classes, num_fp=num_fp)
            else:
                raise Exception('unknown model {}'.format(name))

        elif name == 'reactnet-18':
            print('* Model = ReActNet-18')
            if kwargs['binary_w']:
                print('* Binarize both activation and weights')
                if kwargs['bn_type'] == 'bn':
                    print('* with BN')
                    model = Qaw_reactnet_18_bn(num_classes=num_classes)
                elif kwargs['bn_type'] == 'none':
                    print('* without BN')
                    model = Qaw_reactnet_18_none(num_classes=num_classes)
                elif kwargs['bn_type'] == 'bf':
                    print('* BN-Free')
                    model = Qaw_reactnet_18_bf(num_classes=num_classes)

            else:
                print('* Binarize only activation')
                if kwargs['bn_type'] == 'bn':
                    print('* with BN')
                    model = Qa_reactnet_18_bn(num_classes=num_classes)
                elif kwargs['bn_type'] == 'none':
                    print('* without BN')
                    model = Qa_reactnet_18_none(num_classes=num_classes)
                elif kwargs['bn_type'] == 'bf':
                    print('* BN-Free')
                    model = Qa_reactnet_18_bf(num_classes=num_classes)

        elif name == 'reactnet-A':
            print('* Model = reactnet-A')
            if kwargs['binary_w']:
                print('* Binarize both activation and weights')
                if kwargs['bn_type'] == 'bn':
                    print('* with BN')
                    model = Qaw_reactnet_A_bn(num_classes=num_classes)
                elif kwargs['bn_type'] == 'none':
                    print('* without BN')
                    model = Qaw_reactnet_A_none(num_classes=num_classes)
                elif kwargs['bn_type'] == 'bf':
                    print('* BN-Free')
                    model = Qaw_reactnet_A_bf(num_classes=num_classes)

            else:
                print('* Binarize only activation')
                if kwargs['bn_type'] == 'bn':
                    print('* with BN')
                    model = Qa_reactnet_A_bn(num_classes=num_classes)
                elif kwargs['bn_type'] == 'none':
                    print('* without BN')
                    model = Qa_reactnet_A_none(num_classes=num_classes)
                elif kwargs['bn_type'] == 'bf':
                    print('* BN-Free')
                    model = Qa_reactnet_A_bf(num_classes=num_classes)

        else:
            raise Exception('unknown model {}'.format(name))

        return model
