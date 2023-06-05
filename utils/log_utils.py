import numpy


class LogWriter(object):

    def __init__(self, path, args):
        if '' in args:
            del args['']
        self.path = path
        self.args = args
        with open(self.path, 'w') as f:
            f.write("Training Log\n")
            f.write("Specifications\n")
            for argname in self.args:
                f.write("{} : {}\n".format(argname, self.args[argname]))
            f.write("Checkpoints:\n")

    def checkpoint(self, to_write):
        with open(self.path, 'a') as f:
            f.write(to_write + '\n')

    def initBest(self):
        self.current_best = {
            'loglik': numpy.finfo(float).min,
            'distance': numpy.finfo(float).max,
            'loss': numpy.finfo(float).max,
            'rmse': numpy.finfo(float).max,
            'acc': numpy.finfo(float).min,
        }
        self.episode_best = 'NeverUpdated'

    def updateBest(self, key, value, episode):
        updated = False
        if key == 'loglik' or key == 'acc':
            if value > self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = episode
        elif key == 'distance' or key == 'loss' or key == 'rmse':
            if value < self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = episode
        else:
            raise Exception("unknown key {}".format(key))
        return updated
