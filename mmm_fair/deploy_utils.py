import numpy as np
import os
import shutil
from skl2onnx import to_onnx
import pickle
# from skl2onnx.common.data_types import FloatTensorType  # Optional shape hints


def convert_to_onnx(custom_model, output_path, X, model_type="mmm_fair"):
    """
    Convert each weak estimator in a MMM_Fair ensemble to a separate ONNX file,
    and save additional parameters in a .npy file.
    Then zip up everything into one archive.

    :param custom_model: The MMM_Fair instance (already fitted).
    :param output_path:  Prefix for the generated files. E.g. "my_adult_model"
    :param X:            A sample input array for shape inference.
    """
    # Ensure the model is fitted
    assert (
        len(custom_model.all_estimators) > 0
    ), "Model must be fitted before conversion."

    # Create directory to store ONNX + param files, e.g. "my_adult_model_dir"
    output_path+='/model'
    model_dir = f"{output_path}_{model_type}"
    os.makedirs(model_dir, exist_ok=True)

    # Possibly reduce X for shape inference
    sample_input = X[:1].astype(np.float32)

    # Convert each estimator to ONNX
    if model_type.lower().endswith("gbt"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.utils.validation import check_is_fitted

        clf = HistGradientBoostingClassifier(max_iter=custom_model.max_iter)
        clf._predictors = custom_model.all_estimators
        fitted_attrs = [
            v for v in vars(custom_model) if v.endswith("_") and not v.startswith("__")
        ]
        for v in fitted_attrs:
            clf.__dict__[v] = custom_model.__dict__[v]
        clf.__dict__["_preprocessor"] = custom_model.__dict__["_preprocessor"]
        clf.__dict__["_baseline_prediction"] = custom_model.__dict__[
            "_baseline_prediction"
        ]
        clf.__dict__["_bin_mapper"] = custom_model.__dict__["_bin_mapper"]
        check_is_fitted(clf)
        onnx_file = os.path.join(model_dir, "estimator_gbt.onnx")
        onnx_model = to_onnx(clf, sample_input)
        with open(onnx_file, "wb") as f:
            f.write(onnx_model.SerializeToString())
        params = {
            "fairobs": custom_model.fairobs,
            "all_obs": custom_model.ob,
            "theta": custom_model.theta,
            "sensitives": custom_model.sensitives,
            "pareto": list(custom_model.PF.keys()) if custom_model.PF else [],
        }

    else:
        onnx_models = []
        for i, estimator in enumerate(custom_model.all_estimators):
            onnx_model = to_onnx(estimator, sample_input)
            onnx_models.append(onnx_model)

        # Save each ONNX file in model_dir
        for i, onnx_model in enumerate(onnx_models):
            onnx_file = os.path.join(model_dir, f"estimator_{i}.onnx")
            with open(onnx_file, "wb") as f:
                f.write(onnx_model.SerializeToString())

        # Save additional MMM_Fair parameters in a .npy
        # Adjust keys if needed
        params = {
            "fairobs": custom_model.fairobs,
            "all_obs": custom_model.ob,
            "n_classes": custom_model.n_classes_,
            "classes": custom_model.classes_,
            "alphas": custom_model.estimator_alphas_,
            "theta": custom_model.theta,
            "sensitives": custom_model.sensitives,
            "pareto": list(custom_model.PF.keys()) if custom_model.PF else [],
        }

    params_file = os.path.join(model_dir, "model_params.npy")
    np.save(params_file, params, allow_pickle=True)

    # Finally, zip up the entire directory
    # e.g. creates "my_adult_model_dir.zip"
    archive_name = f"{model_dir}.zip"
    shutil.make_archive(model_dir, "zip", model_dir)

    print(
        f"Saved ONNX models and params to {model_dir}, and archived at {archive_name}"
    )


def convert_to_pickle(custom_model, output_path):
    """
    Serialize the entire MMM_Fair ensemble to a single .pkl file.
    """

    out_file = f"{output_path}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(custom_model, f)
    print(f"Saved pickle model to {out_file}")

from .onnx_utils import ONNX_MMM

def mmm_onnx(path: str = "") -> ONNX_MMM:
    models = []
    model_names = []
    params = None

    def myk(name):
        return int(re.findall(r"[+-]?\d+", name)[0])

    # Read the zip file
    with zipfile.ZipFile(prepare(path)) as myzip:
        # Extract and load the weights file
        for file_name in myzip.namelist():
            if file_name.endswith(".npy"):
                with myzip.open(file_name) as param_file:
                    params = np.load(param_file, allow_pickle=True)
            elif file_name.endswith(".onnx"):
                model_names.append(file_name)

        model_names.sort(key=myk)

        for file_name in model_names:
            with myzip.open(file_name) as model_file:
                model_bytes = model_file.read()
                models.append(model_bytes)
    return ONNX_MMM(models, **dict(params.item()))