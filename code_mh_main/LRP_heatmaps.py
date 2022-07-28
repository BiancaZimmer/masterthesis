import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
import numpy as np

#workaround for innvestigate:
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# import for LRP
import innvestigate
import innvestigate.utils.visualizations as ivis
import helpers_innvestigate

# import from NMNH code
from utils import *
import modelsetup


def methods_grayscale():

    def scale(X):
        return X/255

    def bk_proj(X):
        return ivis.graymap(X)

    def heatmap(X):
        return ivis.heatmap(X)

    def graymap(X):
        return ivis.graymap(np.abs(X), input_is_positive_only=True)

    # Configure analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN            TITLE
        # Show input
        ("input", {}, scale, "Input"),
        # Function
        ("gradient", {"postprocess": "abs"}, graymap, "Gradient"),
        # Signal
        ("guided_backprop", {}, bk_proj, "Guided Backprop"),
        # Interaction
        ("lrp.z", {}, heatmap, "LRP-Z"),
        ("lrp.epsilon", {"epsilon": 1}, heatmap, "LRP-Epsilon"),
    ]
    return methods


if __name__ == '__main__':
    from dataset import DataSet
    from feature_extractor import FeatureExtractor

    dataset_to_use = "mnist"
    suffix_path = "_multicnn"

    print(
        os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'),
        " loading ...")
    model = load_model(
        os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'))

    dataset_used = DataSet(name=dataset_to_use, fe=FeatureExtractor(loaded_model=model))

    methods = methods_grayscale()

    # Create model without trailing softmax
    model_wo_softmax = innvestigate.model_wo_softmax(model)

    model_wo_softmax.get_layer(name='activation')._name = 'activation_ori'
    # for layer in model_wo_softmax.layers:
    #     print(layer.name)

    # Preprocess data
    # data = ( preprocess(x_train), y_train, preprocess(x_test), y_test, )
    ###

    # Create analyzers
    # data eventuell preprocess(x_train) from mnistutils.create_preprocessing_f(x_train, input_range)
    analyzers = helpers_innvestigate.create_analyzers([file.dataentry_to_nparray(use_fe=False) for file in dataset_used.data],
                                      methods, model_wo_softmax)

    n = 2

    test_images = list(zip([file.dataentry_to_nparray(use_fe=False) for file in dataset_used.data_t][:n],
                           [file.ground_truth_label for file in dataset_used.data_t][:n]))

    analysis = np.zeros([len(test_images), len(analyzers), 28, 28, 3])
    text = []

    for i, (x, y) in enumerate(test_images):
        # Add batch axis.
        x = x[None, :, :, :]

        # Predict final activations, probabilities, and label.
        presm = model_wo_softmax.predict_on_batch(x)[0]
        prob = model.predict_on_batch(x)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(
            (
                "%s" % str(y),  # ground truth label
                "%.2f" % presm.max(),  # pre-softmax logits
                "%.2f" % prob.max(),  # probabilistic softmax output
                "%s" % str(y_hat),  # predicted label
            )
        )

        for aidx, analyzer in enumerate(analyzers):
            # Analyze.
            a = analyzer.analyze(x)
            # Apply common postprocessing, e.g., re-ordering the channels for plotting.
            a = helpers_innvestigate.postprocess(a)
            # Apply analysis postprocessing, e.g., creating a heatmap.
            a = methods[aidx][2](a)
            # Store the analysis.
            analysis[i, aidx] = a[0]

    # Prepare the grid as rectangular list
    grid = [
        [analysis[i, j] for j in range(analysis.shape[1])] for i in range(analysis.shape[0])
    ]
    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [
        ("label: {}".format(label[i]), "pred: {}".format(pred[i]))
        for i in range(len(label))
    ]
    row_labels_right = [
        ("logit: {}".format(presm[i]), "prob: {}".format(prob[i]))
        for i in range(len(label))
    ]
    col_labels = ["".join(method[3]) for method in methods]

    # Plot the analysis.
    helpers_innvestigate.plot_image_grid(
        grid,
        row_labels_left,
        row_labels_right,
        col_labels,
        file_name=os.environ.get("plot_file_name", None),
    )

