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
import time, os, sys, numpy as np, shutil as sh
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import json, yaml
import traceback

pjoin = os.path.join


def mkdirs(*paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)


def get_project_path(ExpID):
    full_path = glob.glob('Experiments/*%s*' % ExpID)
    assert (len(full_path) == 1
            )  # There should be only ONE folder with <ExpID> in its name.
    return full_path[0]


class LogPrinter(object):
    """
    Parameters
    ----------
        file (object): .
        ExpID (str): Experiment ID for the logger.
        print_to_screen (bool): Whether to print to screen or not.
    """

    def __init__(self, file, ExpID, print_to_screen=False):
        self.file = file
        self.ExpID = ExpID
        self.print_to_screen = print_to_screen

    def __call__(self, *in_str):
        in_str = [str(x) for x in in_str]
        in_str = ' '.join(in_str)
        short_exp_id = self.ExpID[-6:]
        pid = os.getpid()
        current_time = time.strftime('%Y/%m/%d-%H:%M:%S')
        out_str = '[%s %s %s] %s' % (short_exp_id, pid, current_time, in_str)
        print(out_str, file=self.file, flush=True)  # print to txt
        if self.print_to_screen:
            print(out_str)  # print to screen

    def logprint(self, *in_str):  # to keep the interface uniform
        self.__call__(*in_str)

    def accprint(self, *in_str):
        blank = '  ' * int(self.ExpID[-1])
        self.__call__(blank, *in_str)

    def netprint(self, *in_str):  # i.e., print without any prefix
        """Deprecated.

        Use netprint in Logger.
        """
        for x in in_str:
            print(x, file=self.file, flush=True)
            if self.print_to_screen:
                print(x)

    def print(self, *in_str):
        """Print without any prefix."""
        for x in in_str:
            print(x, file=self.file, flush=True)
            if self.print_to_screen:
                print(x)

    def print_args(self, args):
        """
        Example: ('batch_size', 16) ('CodeID', 12defsd2) ('decoder', models/small16x_ae_base/d5_base.pth)
        It will sort the arg keys in alphabeta order, ignoring the upper/lower difference.
        """
        # build a key map for later sorting
        key_map = {}
        for k in args.__dict__:
            k_lower = k.lower()
            if k_lower in key_map:
                key_map[k_lower + '_' + k_lower] = k
            else:
                key_map[k_lower] = k

        # print in the order of sorted lower keys
        logtmp = ''
        for k_ in sorted(key_map.keys()):
            real_key = key_map[k_]
            logtmp += "('%s': %s) " % (real_key, args.__dict__[real_key])
        self.print(logtmp[:-1] + '\n')  # the last one is blank


class LogTracker(object):
    """
    Parameters
    ----------
        momentum (float): Momentum to be used, default=0.9.
    """

    def __init__(self, momentum=0.9):
        self.loss = OrderedDict()
        self.momentum = momentum
        self.show = OrderedDict()

    def __call__(self, name, value, step=-1, show=True):
        """Update the loss value of <name>"""
        assert type(step) == int
        # value = np.array(value)

        if step == -1:
            if name not in self.loss:
                self.loss[name] = value
            else:
                self.loss[name] = self.loss[name] * self.momentum + value * (
                    1 - self.momentum)
        else:
            if name not in self.loss:
                self.loss[name] = [[step, value]]
            else:
                self.loss[name].append([step, value])

        # if the loss item will show in the log printing
        self.show[name] = show

    def avg(self, name):
        nparray = np.array(self.loss[name])
        return np.mean(nparray[:, 1], axis=0)

    def max(self, name):
        nparray = np.array(self.loss[name])
        # TODO: max index
        return np.max(nparray[:, 1], axis=0)

    def format(self):
        """
        loss example:
            [[1, xx], [2, yy], ...] ==> [[step, [xx, yy]], ...]
            xx ==> [xx, yy, ...]
        """
        keys = self.loss.keys()
        k_str, v_str = [], []
        for k in keys:
            if self.show[k] == False:
                continue
            v = self.loss[k]
            if not hasattr(v, '__len__'):  # xx
                v = '%.4f' % v
            else:
                if not hasattr(v[0], '__len__'):  # [xx, yy, ...]
                    v = ' '.join(['%.3f' % x for x in v])
                elif hasattr(v[0][1], '__len__'):  # [[step, [xx, yy]], ...]
                    v = ' '.join(['%.3f' % x for x in v[-1][1]])
                else:  # [[1, xx], [2, yy], ...]
                    v = '%.4f' % v[-1][1]

            length = min(max(len(k), len(v)), 15)
            format_str = '{:<%d}' % (length)
            k_str.append(format_str.format(k))
            v_str.append(format_str.format(v))
        k_str = ' | '.join(k_str)
        v_str = ' | '.join(v_str)
        return k_str + ' |', v_str + ' |'

    def plot(self, name, out_path):
        """Plot the loss of <name>, save it to <out_path>."""
        v = self.loss[name]
        if (not hasattr(v, '__len__')) or type(
                v[0][0]) != int:  # do not log the 'step'
            return
        if hasattr(v[0][1], '__len__'):
            # self.plot_heatmap(name, out_path)
            return
        v = np.array(v)
        step, value = v[:, 0], v[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(step, value)
        ax.set_xlabel('step')
        ax.set_ylabel(name)
        ax.grid()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def plot_heatmap(self, name, out_path, show_ticks=False):
        """
        A typical case: plot the training process of 10 weights
        x-axis: step
        y-axis: index (10 weights, 0-9)
        value: the weight values
        """
        v = self.loss[name]
        step, value = [], []
        [(step.append(x[0]), value.append(x[1])) for x in v]
        n_class = len(value[0])
        fig, ax = plt.subplots(figsize=[0.1 * len(step),
                                        n_class / 5])  # /5 is set manually
        im = ax.imshow(np.transpose(value), cmap='jet')

        # make a beautiful colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.05, pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # set the x and y ticks
        # For now, this can not adjust its range adaptively, so deprecated.
        # ax.set_xticks(range(len(step))); ax.set_xticklabels(step)
        # ax.set_yticks(range(len(value[0]))); ax.set_yticklabels(range(len(value[0])))

        interval = step[0] if len(step) == 1 else step[1] - step[0]
        ax.set_xlabel('step (* interval = %d)' % interval)
        ax.set_ylabel('index')
        ax.set_title(name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


class Logger(object):
    """The top logger, which (1) set up all log directories (2) maintain the
    losses and accuracies."""

    def __init__(self, args):
        self.args = args

        # set up work folder
        self.ExpID = self.get_ExpID()
        self.Exps_Dir = 'Experiments'
        if hasattr(self.args, 'Exps_Dir'):
            self.Exps_Dir = self.args.Exps_Dir
        self.set_up_dir()

        self.log_printer = LogPrinter(
            self.logtxt, self.ExpID, self.args.debug
            or self.args.screen_print)  # for all txt logging
        self.log_tracker = LogTracker()  # for all numerical logging

        # initial print: save args
        self.print_script()
        self.print_nvidia_smi()
        self.print_note()
        if (not args.debug) and self.SERVER != '':
            # If self.SERVER != '', it shows this is Huan's computer, then call this func, which is just a small feature to my need.
            # When others use this code, they probably need NOT call this func.
            # self.__send_to_exp_hub() # this function is not very useful. deprecated.
            pass
        args.CodeID = self.get_CodeID()
        self.log_printer.print_args(args)
        self.save_args(args)
        self.cache_model()
        self.n_log_item = 0

    def get_CodeID(self):
        if hasattr(self.args, 'CodeID') and self.args.CodeID:
            return self.args.CodeID
        else:
            f = 'wh_git_status_%s.tmp' % time.time()
            script = 'git status >> %s' % f
            os.system(script)
            x = open(f).readlines()
            x = ''.join(x)
            os.remove(f)
            if 'Changes not staged for commit' in x:
                self.log_printer(
                    'Warning! Your code is not committed. Cannot be too careful.'
                )
                time.sleep(3)

            f = 'wh_CodeID_file_%s.tmp' % time.time()
            script = 'git log --pretty=oneline >> %s' % f
            os.system(script)
            x = open(f).readline()
            os.remove(f)
            return x[:8]

    def get_ExpID(self):
        self.SERVER = os.environ['SERVER'] if 'SERVER' in os.environ.keys(
        ) else ''
        TimeID = time.strftime('%Y%m%d-%H%M%S')
        ExpID = 'SERVER' + self.SERVER + '-' + TimeID
        return ExpID

    def set_up_dir(self):
        project_path = pjoin(
            '%s/%s_%s' % (self.Exps_Dir, self.args.project_name, self.ExpID))
        if hasattr(self.args, 'resume_ExpID') and self.args.resume_ExpID:
            project_path = get_project_path(self.args.resume_ExpID)
        if (
                self.args.debug
        ):  # debug has the highest priority. If debug, all the things will be saved in Debug_dir
            project_path = 'Debug_Dir'

        self.exp_path = project_path
        self.weights_path = pjoin(project_path, 'weights')
        self.gen_img_path = pjoin(project_path, 'gen_img')
        self.cache_path = pjoin(project_path, '.caches')
        self.log_path = pjoin(project_path, 'log')
        self.logplt_path = pjoin(project_path, 'log', 'plot')
        self.logtxt_path = pjoin(project_path, 'log', 'log.txt')
        mkdirs(self.weights_path, self.gen_img_path, self.logplt_path,
               self.cache_path)
        self.logtxt = open(self.logtxt_path, 'a+')
        self.script_hist = open(
            '.script_history',
            'a+')  # save local script history, for convenience of check

    def print_script(self):
        script = 'cd %s\n' % os.path.abspath(os.getcwd())
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
            script += ' '.join(
                ['CUDA_VISIBLE_DEVICES=%s python' % gpu_id, *sys.argv])
        else:
            script += ' '.join(['python', *sys.argv])
        script += '\n'
        print(script, file=self.logtxt, flush=True)
        print(script, file=sys.stdout, flush=True)
        print(script, file=self.script_hist, flush=True)

    def print_exc(self):
        traceback.print_exc(file=self.logtxt)

    def print_nvidia_smi(self):
        out = pjoin(self.log_path, 'gpu_info.txt')
        script = 'nvidia-smi >> %s' % out
        os.system(script)

    def print_note(self):
        project = self.get_project_name()  # the current project folder name
        exp_id = self.ExpID.split('-')[-1]  # SERVER138-20200623-095526
        if hasattr(self.args, 'note') and self.args.note:
            self.ExpNote = 'ExpNote [%s-%s-%s]: "%s" -- %s' % (
                self.SERVER,
                project,
                exp_id,
                self.args.note,
                self.args.project_name,
            )
            print(self.ExpNote, file=self.logtxt, flush=True)
            print(self.ExpNote, file=sys.stdout, flush=True)

    def plot(self, name, out_path):
        self.log_tracker.plot(name, out_path)

    def print(self, step):
        keys, values = self.log_tracker.format()
        k = keys.split('|')[0].strip()
        if k:  # only when there is sth to print, print
            values += ' (step = %d)' % step
            if (step % (self.args.print_interval * 10) == 0
                    or len(self.log_tracker.loss.keys()) > self.n_log_item
                ):  # when a new loss is added into the loss pool, print
                self.log_printer(keys)
                self.n_log_item = len(self.log_tracker.loss.keys())
            self.log_printer(values)

    def cache_model(self):
        """Save the model architecture, loss, configs, in case of future
        check."""
        extensions = [
            '.py',
            '.json',
            '.yaml',
            '.sh',
            '.txt',
            '.md',
        ]  # files of these types will be cached
        self.log_printer("==> Caching various config files to '%s'" %
                         self.cache_path)
        for root, dirs, files in os.walk('.'):
            if 'Experiments' in root or 'Debug_Dir' in root:
                continue
            for f in files:
                _, ext = os.path.splitext(f)
                if ext in extensions:
                    dir_path = pjoin(self.cache_path, root)
                    f_path = pjoin(root, f)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    if os.path.exists(f_path):
                        sh.copy(f_path, dir_path)

    def get_project_name(self):
        """For example, 'Projects/CRD/logger.py', then return CRD."""
        file_path = os.path.abspath(__file__)
        return file_path.split('/')[-2]

    def __send_to_exp_hub(self):
        """For every experiment, it will send <ExpNote> to a hub for the
        convenience of checking."""
        today_exp = time.strftime('%Y%m%d') + '_exps.txt'
        if self.SERVER in CONFIDENTIAL_SERVERS:
            today_remote = (
                'huwang@137.203.141.202:/homes/huwang/Projects/ExpLogs/%s' %
                today_exp)
        else:
            today_remote = (
                'wanghuan@155.33.198.138:/home/wanghuan/Projects/ExpLogs/%s' %
                today_exp)
        local_f = 'wh_exps_%s.tmp' % time.time()
        try:
            script_pull = 'scp %s %s' % (today_remote, local_f)
            os.system(script_pull)
        except:
            pass
        with open(local_f, 'a+') as f:
            f.write(self.ExpNote + '\n')
        script_push = 'scp %s %s' % (local_f, today_remote)
        os.system(script_push)
        os.remove(local_f)

    def save_args(self, args):
        with open(pjoin(self.log_path, 'params.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        with open(pjoin(self.log_path, 'params.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f, indent=4)

    def netprint(self, net, comment=''):
        with open(pjoin(self.log_path, 'model_arch.txt'), 'w') as f:
            if comment:
                print('%s:' % comment, file=f)
            print('%s\n' % str(net), file=f, flush=True)


def adjust_learning_rate(optimizer, epoch, num_epochs, scheduler_type, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30
    epochs."""
    if scheduler_type == 1:
        new_lr = lr * (0.5**(epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        if epoch in [num_epochs * 0.5, num_epochs * 0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1


# +
def strlist_to_list(sstr, ttype):
    """
    example:
    # self.args.stage_pr = [0, 0.3, 0.3, 0.3, 0, ]
    # self.args.skip_layers = ['1.0', '2.0', '2.3', '3.0', '3.5', ]
    turn these into a list of <ttype> (float or str or int etc.)
    """
    if not sstr:
        return sstr
    out = []
    sstr = sstr.strip()
    if sstr.startswith('[') and sstr.endswith(']'):
        sstr = sstr[1:-1]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            x = ttype(x)
            out.append(x)
    return out


def strdict_to_dict(sstr, ttype):
    """'{"1": 0.04, "2": 0.04, "4": 0.03, "5": 0.02, "7": 0.03, }'."""
    if not sstr:
        return sstr
    out = {}
    sstr = sstr.strip()
    if sstr.startswith('{') and sstr.endswith('}'):
        sstr = sstr[1:-1]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            k = x.split(':')[0]
            v = ttype(x.split(':')[1].strip())
            out[k] = v
    return out


def check_path(x):
    if x:
        complete_path = glob.glob(x)
        assert len(complete_path) == 1
        x = complete_path[0]
    return x


def merge_args(args, params_json):
    import json, yaml
    """<args> is from argparser.

    <params_json> is a json/yaml file. merge them, if there is collision, the
    param in <params_json> has a higher priority.
    """
    with open(params_json) as f:
        if params_json.endswith('.json'):
            params = json.load(f)
        elif params_json.endswith('.yaml'):
            params = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise NotImplementedError
    for k, v in params.items():
        args.__dict__[k] = v
    return args


# -


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.

    Parameters
    ----------
        decay_schedule (dict): Decay schedule for the LR.
    """

    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = {}
        for (
                k,
                v,
        ) in (decay_schedule.items()
              ):  # a dict, example: {"0":0.001, "30":0.00001, "45":0.000001}
            self.decay_schedule[int(k)] = v
        # print('Using a preset learning rate schedule:')
        # print(self.decay_schedule)

    def __call__(self, optimizer, e):
        epochs = list(self.decay_schedule.keys())
        epochs = sorted(epochs)  # example: [0, 30, 45]
        lr = self.decay_schedule[epochs[-1]]
        for i in range(len(epochs) - 1):
            if epochs[i] <= e < epochs[i + 1]:
                lr = self.decay_schedule[epochs[i]]
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
