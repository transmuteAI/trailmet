class BaseAlgorithm:

    def __init__(self):
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

