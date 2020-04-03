def make_predictions(subject, model_class, params, ph, train, valid, test, weights_file=None, tl_mode="target_training", save_file=None, eval_mode="valid"):
    """
    For every train, valid, test fold, fit the given model with params at prediciton horizon on the training_old set,
    and make predictions on either the validation or testing set
    :param subject: name of subject
    :param model_class: name of model
    :param params: hyperparameters file name
    :param ph: prediction horizon in scaled minutes
    :param train: training_old sets
    :param valid: validation sets
    :param test: testing sets
    :param mode: on which set the model is tested (either "valid" or "test")
    :return: array of ground truths/predictions dataframe
    """
    results = []
    for i, (train_i, valid_i, test_i) in enumerate(zip(train, valid, test)):
        model = model_class(subject, ph, params, train_i, valid_i, test_i)
        model.fit(weights_file, tl_mode, save_file)
        res = model.predict(dataset=eval_mode)
        results.append(res)
        if eval_mode == "valid":
            break
    return results
