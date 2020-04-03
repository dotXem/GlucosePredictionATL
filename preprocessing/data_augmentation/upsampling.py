def upsample_by_target_nan_filling(data):
    y_interpolated = data.y.interpolate("linear")
    data.y = y_interpolated
    return data