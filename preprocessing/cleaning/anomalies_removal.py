import numpy as np
from tools.printd import printd

def remove_anomalies(data, anomalies_threshold=2.5, plot=False):
    data_cp = data.copy()
    for i in range(5):
        anomalies_indexes = detect_glucore_readings_anomalies(data_cp, "datetime", "glucose",
                                                              threshold=anomalies_threshold)
        data_cp = data_cp.drop(anomalies_indexes, axis=0)
        data_cp = data_cp.reset_index(drop=True)
        if plot:
            printd("[iter {}] Number of anomalies removed : {}".format(i, len(anomalies_indexes)))

    if plot:
        ax=data.plot("datetime","glucose")
        data_cp.plot("datetime","glucose",ax=ax)
        import matplotlib.pyplot as plt
        plt.show()



    return data_cp


def detect_glucore_readings_anomalies(df, datetime_col,glucose_col, threshold):
    df_nonan = df.drop(["CHO","insulin"],axis=1).dropna()
    i = df_nonan.index
    t = df_nonan[datetime_col].astype(np.int64).values
    g = df_nonan[glucose_col].values

    arr = np.concatenate([g.reshape(-1, 1), t.reshape(-1, 1)], axis=1)

    # compute the contexts
    contexts = [arr[i-1:i + 2] for i in range(1,len(arr) - 2)]

    # compute the variations
    # variations = np.array([np.divide(np.diff(context[:, 1]), np.diff(context[:, 0])) for context in contexts])
    variations = np.diff(contexts, axis=1)
    variations = np.divide(variations[:,:,0], variations[:,:,1])

    # compute the behavior of the variations
    mean = np.nanmean(variations)
    std = np.nanstd(variations)

    # compute z_score
    z_score = np.divide(np.subtract(variations, mean), std)

    # flag variations that are anomalies
    anomalies_indexes = np.array(np.where(np.all(
        np.c_[np.abs(z_score)[:, 0] >= threshold, np.abs(z_score)[:, 1] >= threshold, np.prod(z_score, axis=1) < 0],
        axis=1))).reshape(-1, 1) + 1

    return i[anomalies_indexes].ravel()