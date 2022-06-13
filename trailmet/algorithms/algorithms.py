class BaseAlgorithm:

    def __init__(self):

        # set default value for parameters generic to all th algorithms

        pass

    def compress_model(self):
        pass

    def set_hyperparams(device=None,
                        dataset=None,
                        optimizer='SGD',
                        scheduler=None,
                        ):
        self.optimizer = optimizer
        pass

