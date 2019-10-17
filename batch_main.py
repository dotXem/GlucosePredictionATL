from main import main_target, main_source
from models.FCN import FCN, params as fcn_params

def batch_main():
    source_dataset = "IDIAB"
    target_dataset = "IDIAB"
    target_subjects = ["1","2","3","4","5"]
    model, params = FCN, fcn_params
    eval = "valid"
    weights = "lambda017"
    save = "lambda017_noL2_lr4_valid"
    mode = "target_global"

    for target_subject in target_subjects:
        main_target(mode=mode,
                    source_dataset=source_dataset,
                    target_dataset=target_dataset,
                    target_subject=target_subject,
                    Model=model,
                    params=params,
                    weights_dir=weights,
                    eval=eval,
                    split=None,
                    save_file=save,
                    plot=False)

if __name__ == "__main__":
    batch_main()