import sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa


class iris_model:
    def __init__(self, iris_csv):
        self.load_dataset(iris_csv)
        # SciKit-Learn API compliant model

    def load_dataset(self, iris_csv):
        self.data = pd.read_csv(iris_csv, header=None)

        # Load the Dataset from file into X (features) and encoded Y (targets)
        self.y = self.data.iloc[:, -1]
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        self.x  = self.data.iloc[:, 0:-1]

    def train_model(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.1)
        self.pipe = Pipeline([('scalar', StandardScaler()),
                             ('xgb', xgb.XGBClassifier(n_estimators=3))]
                             )
        self.name = 'XGBoost'

        # Fit & Score the Model
        self.pipe.fit(self.x_train, self.y_train)
        self.y_pred = self.pipe.predict(self.x_test)
        self.score = sklearn.metrics.accuracy_score(self.y_test, self.y_pred)

    def inference_model(self, x):
        x = np.array(x)
        x = x.reshape(1, -1)
        return self.pipe.predict(x)

    def get_score(self):
        return self.score

    def save_onnx(self):
        update_registered_converter(
            XGBClassifier, 'XGBoostXGBClassifier',
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

    # Convert to ONNX format
        self.model_onnx = convert_sklearn(
            self.pipe, 'pipeline_xgboost',
            [('input', FloatTensorType([None, 4]))],
            target_opset={'': 12, 'ai.onnx.ml': 2})

    # Write ONNX Model to file
        with open('pipeline_xgboost.onnx', "wb") as f:
            f.write(self.model_onnx.SerializeToString())

    def get_onnx_graph(self):
        # Create Graph (Look at this graph!)
        pydot_graph = GetPydotGraph(
            self.model_onnx.graph,
            name=self.model_onnx.graph.name, rankdir="TB",
            node_producer=GetOpNodeProducer(
                "docstring", color="navy",
                fillcolor="yellow", style="filled"))
        pydot_graph.write_dot("pipeline.dot")

        os.system('dot -O -Gdpi=300 -Tpng pipeline.dot')

        image = plt.imread("pipeline.dot.png")
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.imshow(image)
        ax.axis('off')


def main():
    my_iris = iris_model('iris.csv')
    my_iris.train_model()
    my_iris.save_onnx()
    print("Model Trained with Score: ", my_iris.get_score())
    my_iris.get_onnx_graph()
    test_value = np.array([6.8, 3.0, 5.5, 2.1])
    test_value = test_value.reshape(1, -1)
    print("Predict", my_iris.inference_model(test_value))


if __name__ == "__main__":
    main()
