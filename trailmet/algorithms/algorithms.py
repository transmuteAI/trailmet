class BaseAlgorithm:

    def __init__(self):

        # set default value for parameters generic to all th algorithms

        pass

    def compress_model(self):
        pass

    def pretrain_epoch(self, model, loss_fn, optimizer, scheduler = None):
        model.train()
        counter = 0
        tk1 = tqdm_notebook(dataloaders['train'], total=len(dataloaders['train']))
        running_loss = 0
        for x_var, y_var in tk1:
            counter +=1
            x_var = x_var.to(device=device)
            y_var = y_var.to(device=device)
            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            running_loss+=loss.item()
            tk1.set_postfix(loss=running_loss/counter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return running_loss/counter

    def set_hyperparams(device=None,
                        dataset=None,
                        optimizer='SGD',
                        scheduler=None,
                        ):
        self.optimizer = optimizer
        pass

