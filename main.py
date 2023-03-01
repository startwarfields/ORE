import sklearn
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa


def main():
    data = pd.read_csv('iris.csv', header=None)

    # Load the Dataset from file into X (features) and encoded Y (targets)
    y = data.iloc[:, -1]
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)
    x = data.iloc[:, 0:-1]

    # Train Test Split &  Load the pipeline
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.1)
    pipe = Pipeline([('scalar', StandardScaler()),
                     ('lgbm', xgb.XGBClassifier(n_estimators=3))
                     ])
    name = 'XGBoost'

    # Fit & Score the Model
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    score = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(name, " has accuracy of:", "%0.2f" % score)

    # Update container to register xgboost
    update_registered_converter(
        XGBClassifier, 'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})

    # Convert to ONNX format
    model_onnx = convert_sklearn(
        pipe, 'pipeline_xgboost',
        [('input', FloatTensorType([None, 4]))],
        target_opset={'': 12, 'ai.onnx.ml': 2})

    # Write ONNX Model to file
    with open('pipeline_xgboost.onnx', "wb") as f:
        f.write(model_onnx.SerializeToString())

    # Create Graph (Look at this graph!)
    pydot_graph = GetPydotGraph(
        model_onnx.graph, name=model_onnx.graph.name, rankdir="TB",
        node_producer=GetOpNodeProducer(
            "docstring", color="navy",
            fillcolor="yellow", style="filled"))
    pydot_graph.write_dot("pipeline.dot")

    os.system('dot -O -Gdpi=300 -Tpng pipeline.dot')

    image = plt.imread("pipeline.dot.png")
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.imshow(image)
    ax.axis('off')


if __name__ == "__main__":
    main()
