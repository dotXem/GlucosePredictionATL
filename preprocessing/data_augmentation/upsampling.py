def upsample_by_target_nan_filling(data):
    """
    Linearly upsample the objective glucose prediction. Should not be used on test set, as it is illegal.
    :param data: samples Dataframe
    :return: samples Dataframe with modified y
    """
    y_interpolated = data.y.interpolate("linear")
    data.y = y_interpolated
    return data