import gradio as gr
import iris


class inference_wrapper:
    def __init__(self):
        self.live_iris = iris.iris_model('iris.csv')
        self.live_iris.train_model()

    def inference(self, sepal_width, sepal_length, petal_width, petal_length) -> str:
        x = [sepal_width, sepal_length, petal_width, petal_length]
        prediction = self.live_iris.inference_model(x)
        return self.convert_num_to_class(prediction)

    def convert_num_to_class(self,num: int) -> str:
        classes = {0: "Setosa",
               1: "Versicolor",
               2:  "Virginica"}
        return classes[num[0]]


def main():
    live_inf = inference_wrapper()
    demo = gr.Interface(fn=live_inf.inference, inputs=[gr.Slider(0,10), gr.Slider(0,10), gr.Slider(0,10), gr.Slider(0,10)],
                        outputs="text")
    demo.launch()


if __name__ == "__main__":
    main()


