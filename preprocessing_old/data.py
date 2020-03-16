class Data():
    def __init__(self, train, valid, test, mean, std):
        self.train = train
        self.valid = valid
        self.test = test
        self.mean = mean
        self.std = std

    def get_split(self, split_number):
        return self._get_split_x_y(self.train, split_number), self._get_split_x_y(self.valid, split_number), \
               self._get_split_x_y(self.test, split_number), self.mean[split_number], self.std[split_number]

    def _get_split_x_y(self, xy, split_number):
        return [xy[0][split_number], xy[1][split_number]]
