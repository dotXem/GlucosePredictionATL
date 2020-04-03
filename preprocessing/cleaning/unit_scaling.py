import numpy as np

def scaling_T1DMS(data):
    # scale insulin from pmol to unit
    data.loc[:, "insulin"] = data.loc[:, "insulin"] / 6000.0

    # accumulate the CHO intakes
    CHO_indexes = data[np.invert(data.loc[:, "CHO"] == 0.0)].index
    meals, meal, start_idx, past_idx = [], data.loc[CHO_indexes[0],"CHO"], CHO_indexes[0], CHO_indexes[0]
    for idx in CHO_indexes[1:]:
        if idx == past_idx+1:
            meal = meal + data.loc[idx, "CHO"]
        else:
            meals.append([start_idx, meal])
            meal = data.loc[idx, "CHO"]
            start_idx = idx
        past_idx = idx
    meals.append([start_idx, meal])
    meals = np.array(meals)

    data.loc[:, "CHO"] = 0.0
    data.loc[meals[:,0],"CHO"] = meals[:,1]

    return data