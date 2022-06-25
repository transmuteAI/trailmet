def adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if scheduler_type==1:
        new_lr = lr * (0.5 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        if epoch in [num_epochs*0.5, num_epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1