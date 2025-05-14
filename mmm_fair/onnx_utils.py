import onnxruntime as rt
import numpy as np
import re
import numpy as np
import zipfile

class ONNX_MMM():
    def __init__(
        self,
        models,
        alphas=None,
        classes=None,
        n_classes=None,
        theta=None,
        pareto=None,
        sensitives=None,
    ):
        self.models = models
        self.pareto = pareto
        self.alphas = alphas
        self.sensitive = sensitives
        self.classes = classes
        self.theta = theta
        self.n_classes = n_classes

    def predict(self, dataset, sensitive, theta=None):
        """assert (
            sensitive is None or len(sensitive) == 0
        ), "ONNXEnsemble can only be called with no declared sensitive attributes" """
        theta = theta if self.theta is None else self.theta
        sensitive = sensitive if self.sensitive is None else self.sensitive
        X = dataset.to_pred(sensitive)
        X = X.astype(np.float32)
        if len(self.models)>1:
            try:
                classes = self.classes[:, np.newaxis]
        
                pred = 0
                i = 0
                for estimator, alpha in zip(
                    self.models[:theta],
                    self.alphas[:theta],
                ):
                    #notify_progress(i / theta, f"Running ensemble voter {i}/{theta}")
                    session = rt.InferenceSession(estimator)
                    input_name = session.get_inputs()[0].name
                    pred += (session.run(None, {input_name: X})[0] == classes).T * alpha
                    i += 1
                #notify_progress(1, f"Completed ensemble voting")
                pred /= self.alphas[:theta].sum()
                pred[:, 0] *= -1
                preds = classes.take(pred.sum(axis=1) > 0, axis=0)
                return np.squeeze(preds, axis=1)   
            except Exception as e:
                print("ONNX predict failed:", e)
                raise ValueError(f'Inputs or params mismatch for mmm ensemble: {e}')
        else:
            try:
                model_byte=self.models[0]
                session = rt.InferenceSession(model_bytes)
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: X})
                return outputs[0]
            except Exception as e:
                print("ONNX predict failed:", e)
                raise ValueError(f'Inputs or params mismatch: {e}')