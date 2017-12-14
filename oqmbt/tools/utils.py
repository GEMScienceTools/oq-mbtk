import numpy


class GetSourceIDs(object):

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.keys = set([key for key in self.model.sources])

    def keep_equal_to(self, param_name, values):
        """
        :parameter str param_name:
        :parameter list values:
        """
        assert type(values) is list
        tmp = []
        for key in self.keys:
            src = self.model.sources[key]
            param_value = getattr(src, param_name)
            for value in values:
                if param_value == value:
                    tmp.append(key)
                    continue
        self.keys = tmp

    def keep_gt_than(self, param_name, value):
        tmp = []
        for key in self.keys:
            src = self.model.sources[key]
            param_value = getattr(src, param_name)
            if param_value > value:
                tmp.append(key)
        self.keys = tmp

    def keep_lw_than(self, param_name, value):
        tmp = []
        for key in self.keys:
            src = self.model.sources[key]
            param_value = getattr(src, param_name)
            if param_value < value:
                tmp.append(key)
        self.keys = tmp
