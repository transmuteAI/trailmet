from .resnet import get_resnet_model


class ModelsFactory(object):
    @staticmethod
    def create_model(name, num_classes=100, pretrained = False, **kwargs):
        """
        Returns the requested model, ready for training/pruning with the specified method

        Args:
            name: model name 'resnet18','resnet50'
            num_classes: number of classes
        Return:
            model object
        """

        if 'resnet' in name:
            assert 'insize' in kwargs, "should provide input size"
            insize = kwargs['insize']
            model = get_resnet_model(name, num_classes, insize, pretrained = pretrained)
        else:
            raise Exception("unknown model {}".format(kwargs['name']))
        return model