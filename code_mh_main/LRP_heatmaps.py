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
from modelsetup import *


def bk_proj(X):
    return ivis.graymap(X)


def heatmap(X):
    return ivis.heatmap(X)


def graymap(X):
    return ivis.graymap(np.abs(X), input_is_positive_only=True)


def methods_grayscale():

    def scale(X):
        return X

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
        ("lrp.w_square", {}, heatmap, "LRPWSquare"),
        ("lrp.flat", {}, heatmap, "LRPFlat"),
        ("lrp.alpha_beta", {"alpha": 3, "beta": 2}, heatmap, "LRPAlphaBeta"),
        ("lrp.alpha_2_beta_1_IB", {}, heatmap, "LRPAlpha2Beta1IgnoreBias"),
        ("lrp.alpha_1_beta_0", {}, heatmap, "LRPAlpha1Beta0"),
        ("lrp.alpha_1_beta_0_IB", {}, heatmap, "LRPAlpha1Beta0IgnoreBias"),
        ("lrp.z_plus", {}, heatmap, "LRPZPlus"),
        ("lrp.z_plus_fast", {}, heatmap, "LRPZPlusFast"),
        ("lrp.sequential_preset_a", {"epsilon": 0.1}, heatmap, "LRPSequentialPresetA"),
        ("lrp.sequential_preset_b", {"epsilon": 0.11}, heatmap, "LRPSequentialPresetB"),
        ("lrp.sequential_preset_a_flat", {}, heatmap, "LRPSequentialPresetAFlat"),
        ("lrp.sequential_preset_b_flat", {}, heatmap, "LRPSequentialPresetBFlat"),
        ("lrp.sequential_preset_b_flat_until_idx", {}, heatmap, "LRPSequentialPresetBFlatUntilIdx"),
    ]
    return methods


def methods_mnist():

    def scale(X):
        return X

    # Configure analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN            TITLE
        # Show input
        ("input", {}, scale, "Input"),
        # Function
        ("gradient", {"postprocess": "abs"}, graymap, "Gradient"),
        # Signal
        ("guided_backprop", {}, bk_proj, "Guided Backprop"),
        # # Interaction
        ("lrp.sequential_preset_a", {"epsilon": 0.1}, heatmap, "LRPSequentialPresetA"),
        ("lrp.sequential_preset_b", {"epsilon": 0.11}, heatmap, "LRPSequentialPresetB"),
        ("lrp.sequential_preset_a_flat", {}, heatmap, "LRPSequentialPresetAFlat"),
        ("lrp.sequential_preset_b_flat", {}, heatmap, "LRPSequentialPresetBFlat"),
        ("lrp.sequential_preset_b_flat_until_idx", {}, heatmap, "LRPSequentialPresetBFlatUntilIdx"),
    ]
    return methods


def methods_oct():

    def scale(X):  # scale data between [0,1]
        a, b = X.min(), X.max()
        c, d = [0, 1]
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= b - a
        X *= d - c
        # shift to desired output range
        X += c
        return X

    # Configure analysis methods and properties
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN            TITLE
        # Show input
        ("input", {}, scale, "Input"),
        # # Interaction
        ("lrp.alpha_1_beta_0", {}, heatmap, "LRPAlpha1Beta0"),
        ("lrp.sequential_preset_a", {"epsilon": 0.1}, heatmap, "LRPSequentialPresetA"),  # epsilon?
        ("lrp.sequential_preset_a_flat", {}, heatmap, "LRPSequentialPresetAFlat"),
        ("lrp.sequential_preset_b_flat_until_idx", {}, heatmap, "LRPSequentialPresetBFlatUntilIdx")
    ]
    return methods


def generate_method_comparison(dataset_to_use, suffix_path, type_of_model, methods, number_images=10):

    print(
        os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'),
        " loading ...")
    model = load_model(
        os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'))

    if type_of_model == "vgg":
        feature_model_output_layer = model.get_layer('flatten').output
    else:
        feature_model_output_layer = model.layers[-3].output

    setup_model = train_eval_model(dataset_to_use, fit=False, type_of_model=type_of_model, suffix_path=suffix_path,
                                   model_for_feature_embedding=model,
                                   eval=False, loss=False, missclassified=False,
                                   feature_model_output_layer=feature_model_output_layer)
    print("You are using a ", setup_model.dataset.fe.fe_model.name)
    train_data = setup_model.dataset.data
    x_test = [file.dataentry_to_nparray(use_fe=False) for file in setup_model.dataset.data_t]
    y_test = [file.ground_truth_label for file in setup_model.dataset.data_t]

    rand_idx = [random.randint(0, len(y_test)) for p in range(0, number_images)]
    test_images = list(zip([x_test[i] for i in rand_idx],
                           [y_test[i] for i in rand_idx]))

    # Create model without trailing softmax
    model_wo_softmax = innvestigate.model_wo_softmax(model)

    try:
        model_wo_softmax.get_layer(name='activation')._name = 'activation_ori'
    except ValueError:
        pass
    # for layer in model_wo_softmax.layers:
    #     print(layer.name)

    # Create analyzers
    analyzers = helpers_innvestigate.create_analyzers(
        [file.dataentry_to_nparray(use_fe=False) for file in train_data],
        methods,
        model_wo_softmax,
        "max_activation"
    )
    analysis = np.zeros([len(test_images), len(analyzers), setup_model.img_size, setup_model.img_size, 3])
    text = []

    for i, (x, y) in enumerate(test_images):
        # Add batch axis.
        x = x[None, :, :, :]
        print("Computing LRP Heatmaps for image #", i, " ...")
        # print(np.shape(x))

        # Predict final activations, probabilities, and label.
        presm = model_wo_softmax.predict_on_batch(x)[0]
        prob = model.predict_on_batch(x)[0]
        y_hat = np.argmax(prob)  # prob.argmax()
        if not BINARY:
            y_hat = setup_model.labelencoder.inverse_transform([y_hat])

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
        (f"label: {label[i]}", f"pred: {pred[i]}") for i in range(len(label))
    ]
    row_labels_right = [
        (f"logit: {presm[i]}", f"prob: {prob[i]}") for i in range(len(label))
    ]
    col_labels = ["".join(method[3]) for method in methods]

    # Plot the analysis.
    helpers_innvestigate.plot_image_grid(
        grid, row_labels_left, row_labels_right, col_labels,
        file_name=os.environ.get("plot_file_name", None),
    )


def generate_method_and_neuron_comparison(dataset_to_use, suffix_path, type_of_model, methods, number_images=2):

    print(
        os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'),
        " loading ...")
    model = load_model(
        os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'))

    if type_of_model == "vgg":
        feature_model_output_layer = model.get_layer('flatten').output
    else:
        feature_model_output_layer = model.layers[-3].output

    setup_model = train_eval_model(dataset_to_use, fit=False, type_of_model=type_of_model, suffix_path=suffix_path,
                                   model_for_feature_embedding=model,
                                   eval=False, loss=False, missclassified=False,
                                   feature_model_output_layer=feature_model_output_layer)
    print("You are using a ", setup_model.dataset.fe.fe_model.name)
    train_data = setup_model.dataset.data
    x_test = [file.dataentry_to_nparray(use_fe=False) for file in setup_model.dataset.data_t]
    y_test = [file.ground_truth_label for file in setup_model.dataset.data_t]

    rand_idx = [random.randint(0, len(y_test)) for p in range(0, number_images)]
    test_images = list(zip([x_test[i] for i in rand_idx],
                           [y_test[i] for i in rand_idx]))
    num_classes = len(setup_model.labelencoder.classes_)

    # Create model without trailing softmax
    model_wo_softmax = innvestigate.model_wo_softmax(model)

    try:
        model_wo_softmax.get_layer(name='activation')._name = 'activation_ori'
    except ValueError:
        pass
    # for layer in model_wo_softmax.layers:
    #     print(layer.name)

    # Create analyzers
    analyzers = helpers_innvestigate.create_analyzers(
        [file.dataentry_to_nparray(use_fe=False) for file in train_data],
        methods,
        model_wo_softmax,
        "index"
    )

    for image_nr, (x, y) in enumerate(test_images):
        # Add batch axis.
        x = x[None, :, :, :]
        print("Computing LRP Heatmaps for image #", image_nr, " ...")

        analysis = np.zeros([num_classes, len(analyzers), setup_model.img_size, setup_model.img_size, 3])
        text = []

        for ii, output_neuron in enumerate(range(num_classes)):
            # Predict final activations, probabilites, and label.
            presm = model_wo_softmax.predict_on_batch(x)[0]
            prob = model.predict_on_batch(x)[0]
            output_neuron_label = output_neuron
            if not BINARY:
                output_neuron_label = setup_model.labelencoder.inverse_transform([output_neuron])

            # Save prediction info:
            text.append(
                (
                    "%s" % str(y),  # ground truth label
                    "%.2f" % presm[output_neuron],  # pre-softmax logits
                    "%.2f" % prob[output_neuron],  # probabilistic softmax output
                    "%s" % str(output_neuron_label)
                )
            )

            for aidx, analyzer in enumerate(analyzers):
                # Analyze.
                a = analyzer.analyze(x, neuron_selection=output_neuron)
                # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                a = helpers_innvestigate.postprocess(a)
                # Apply analysis postprocessing, e.g., creating a heatmap.
                a = methods[aidx][2](a)
                # Store the analysis.
                analysis[ii, aidx] = a[0]

        # Prepare the grid as rectangular list
        grid = [
            [analysis[i, j] for j in range(analysis.shape[1])] for i in range(analysis.shape[0])
        ]
        # Prepare the labels
        label, presm, prob, pred = zip(*text)
        row_labels_left = [
            (f"label: {label[i]}", f"pred: {pred[i]}") for i in range(len(label))
        ]
        row_labels_right = [
            (f"logit: {presm[i]}", f"prob: {prob[i]}") for i in range(len(label))
        ]
        col_labels = ["".join(method[3]) for method in methods]

        # Plot the analysis.
        file_name = os.environ.get("PLOTFILENAME", None)
        if file_name is not None:
            file_name = (
                    ".".join(file_name.split(".")[:-1])
                    + ("_%i" % output_neuron)
                    + file_name.split(".")[-1]
            )
        helpers_innvestigate.plot_image_grid(
            grid, row_labels_left, row_labels_right, col_labels, file_name=file_name
        )


if __name__ == '__main__':
    from dataset import DataSet
    from feature_extractor import FeatureExtractor
    from modelsetup import ModelSetup

    # generate_method_comparison(dataset_to_use="mnist", suffix_path="_multicnn", type_of_model="cnn",
    #                            methods=methods_mnist(), number_images = 10)

    # generate_method_comparison(dataset_to_use="oct_small_cc", suffix_path="_vgg", type_of_model="vgg",
    #                            methods=methods_oct(), number_images=10)

    generate_method_and_neuron_comparison(dataset_to_use="mnist", suffix_path="_multicnn", type_of_model="cnn",
                                          methods=methods_mnist(), number_images=1)

    # generate_method_and_neuron_comparison(dataset_to_use="oct_small_cc", suffix_path="_vgg", type_of_model="vgg",
    #                                       methods=methods_oct(), number_images=3)

    # possible inputs for "methods":
    # methods_grayscale()   for grayscale data
    # methods_mnist()       for dataset mnist
    # methods_oct()         for datasets oct

    # TODO: function to calculate + store LRP Heatmaps
    # TODO: function to compare neuron outputs on x-Axis, images on y-Axis



