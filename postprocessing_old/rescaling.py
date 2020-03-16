
def rescale(y, mean, std):
    return y * std[-1] + mean[-1]