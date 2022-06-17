

from ..algorithms import BaseAlgorithm


class BasePruning(BaseAlgorithm):

    def __init__(self):
        super(BasePruning, self).__init__()

        # Set default values of parameters generic to all pruning methods
        self.epochs = {'pretrain': 50, 'prune': 20, 'finetune': 20}
        self.optimizer = 'Adam'

        pass

    def pretrain(self):
        best_acc = 0
        num_epochs = args.epochs
        train_losses = []
        valid_losses = []
        valid_accuracy = []
        if args.test_only == False:
            for epoch in range(num_epochs):
                adjust_learning_rate(optimizer, epoch, args)
                print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
                t_loss = train(model, criterion, optimizer)
                acc, v_loss = test(model, criterion, optimizer, "val")

                if acc>best_acc:
                    print("**Saving model**")
                    best_acc=acc
                    torch.save({
                        "epoch": epoch + 1,
                        "state_dict" : model.state_dict(),
                        "acc" : best_acc,
                    }, f"checkpoints/{args.model}_{args.dataset}_pretrained.pth")

                train_losses.append(t_loss)
                valid_losses.append(v_loss)
                valid_accuracy.append(acc)
                df_data=np.array([train_losses, valid_losses, valid_accuracy]).T
                df = pd.DataFrame(df_data, columns = ['train_losses','valid_losses','valid_accuracy'])
                df.to_csv(f'logs/{args.model}_{args.dataset}_pretrained.csv')

    def prune(self)
