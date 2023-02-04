import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from matplotlib import pyplot as plt

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
    # creating a heatmap - other color types:
    # "bwr" (lighter in color than seismic)
    # "coolwarm" (grey instead of white as neutral value)
    # "seismic"
    # binary or greymap or gray are white to black or black to white
    return ivis.heatmap(X, cmap_type="seismic")


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
        # Function
        # ("gradient", {"postprocess": "abs"}, graymap, "Gradient"),  # not working with Batch Normalization
        # Signal
        # ("guided_backprop", {}, bk_proj, "Guided Backprop"), # not working with Batch Normalization
        # Interaction
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

    model = None
    feature_model_output_layer = None

    if suffix_path != "":
        print(
            os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'),
            " loading ...")
        model = load_model(
            os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'))

        feature_model_output_layer = get_output_layer(model, type_of_model)

    setup_model = train_eval_model(dataset_to_use, fit=False, type_of_model=type_of_model, suffix_path=suffix_path,
                                   model_for_feature_embedding=model,
                                   eval=False, loss=False, misclassified=False,
                                   feature_model_output_layer=feature_model_output_layer)
    print("You are using a ", setup_model.dataset.fe.fe_model.name)
    train_data = setup_model.dataset.data

    # if you only want to select images from a certain class, use this line instead of the next
    # xt = [file for file in setup_model.dataset.data_t if file.ground_truth_label == "DRUSEN"]
    xt = setup_model.dataset.data_t
    rand_idx = random.sample(range(0, len(xt)), number_images)
    x_test = [setup_model.img_preprocess_for_prediction(xt[i]) for i in rand_idx]
    y_test = [xt[i].ground_truth_label for i in rand_idx]

    test_images = list(zip([i for i in x_test],
                           [i for i in y_test]))

    # Create model without trailing softmax
    model_wo_softmax = innvestigate.model_wo_softmax(setup_model.model)

    try:
        model_wo_softmax.get_layer(name='activation')._name = 'activation_ori'
    except ValueError:
        pass
    # for layer in model_wo_softmax.layers:
    #     print(layer.name)

    # Create analyzers
    analyzers = helpers_innvestigate.create_analyzers(
        methods,
        model_wo_softmax,
        "max_activation"
    )
    analysis = np.zeros([len(test_images), len(analyzers), setup_model.img_size, setup_model.img_size, 3])
    text = []

    for i, (x, y) in enumerate(test_images):
        # Add batch axis.
        # x = x[None, :, :, :]
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

    feature_model_output_layer = get_output_layer(model, type_of_model)

    setup_model = train_eval_model(dataset_to_use, fit=False, type_of_model=type_of_model, suffix_path=suffix_path,
                                   model_for_feature_embedding=model,
                                   eval=False, loss=False, misclassified=False,
                                   feature_model_output_layer=feature_model_output_layer)
    print("You are using a ", setup_model.dataset.fe.fe_model.name)
    train_data = setup_model.dataset.data
    x_test = [setup_model.img_preprocess_for_prediction(file) for file in setup_model.dataset.data_t]
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
        methods,
        model_wo_softmax,
        "index"
    )

    for image_nr, (x, y) in enumerate(test_images):
        # Add batch axis.
        # x = x[None, :, :, :]
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
                print("Using method ", methods[aidx][3])
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


def create_special_analyzer(model, dataset_to_use):
    """
    Creates analyzers for special data sets
    :param model: A trained CNN model, if None an untrained VGG16 is used
    :param dataset_to_use: str, name of the data set
    :return: analyzer according to the innvestigate analyzer class
    """
    if 'mnist' in dataset_to_use:
        method = 'lrp.sequential_preset_a'
        parameters = {'epsilon': 0.1}
    elif 'oct' in dataset_to_use:
        method = 'lrp.sequential_preset_a_flat'
        parameters = {}
    else:
        print("CAREFUL! No best practice analyzer defined for this dataset")
        method = 'lrp.sequential_preset_a'
        parameters = {}

    # Create model without trailing softmax for analyzer
    if model is None:
        model = VGG16(weights='imagenet', include_top=True)
        model_wo_softmax = innvestigate.model_wo_softmax(model)
    else:
        try:
            model.get_layer(name='dense')._name = 'dense_ori'
        except ValueError:
            pass  # when there is no layer named "dense" you don't need to rename it
        model_wo_softmax = innvestigate.model_wo_softmax(model)
    try:
        model_wo_softmax.get_layer(name='activation')._name = 'activation_ori'
    except ValueError:
        pass  # when there is no layer named "activation" you don't need to rename it

    # Set up analyzer
    analyzer = innvestigate.create_analyzer(
        method,  # analysis method identifier as str eg "lrp.sequential_preset_a"
        model_wo_softmax,  # model without softmax output
        neuron_selection_mode="index",
        **parameters  # as dictionary eg {"epsilon": 0.1}
    )
    return analyzer


def generate_LRP_heatmap(x, analyzer, output_neuron):
    """ Generates one heatmap for one input image

    :param x: image as numpy array
    :param analyzer: innvestigate analyzer
    :param output_neuron: neuron (label) for which to output the heatmap
    :return: heatmap as numpy array
    """
    # Add batch axis.
    # x = x[None, :, :, :]
    # Analyze.
    a = analyzer.analyze(x, neuron_selection=output_neuron)
    # Apply common postprocessing, e.g., re-ordering the channels for plotting.
    a = helpers_innvestigate.postprocess(a)
    # creating a heatmap - other color types:
    # "bwr" (lighter in color than seismic
    # "coolwarm" (grey instead of white as neutral value)
    # "seismic"
    # binary or greymap or gray are white to black or black to white
    a = ivis.heatmap(a, cmap_type='binary')

    return a[0]


def generate_LRP_heatmaps_for_dataset(dataset_to_use, suffix_path, type_of_model, method, parameters={}, base_vgg=False, save=True):
    """ Creates (and saves) LRP heatmaps for every image in the test and train set of a given dataset. This is done for every possible output neuron/label

    :param dataset_to_use: name of dataset
    :type dataset_to_use: str
    :param suffix_path: suffix of model of this data set for which the heatmaps should be calculated
    :type suffix_path: str
    :param type_of_model: one of cnn/vgg/inception; same as for modelsetup.train_eval_model()
    :param method: analysis method identifier for the LRP analyzer eg "lrp.sequential_preset_a"
    :type method: str
    :param parameters: optional parameters for the LRP analyzer. If none put empty dictionary {} else eg {"epsilon": 0.1}
    :type parameters: dict
    :param base_vgg: create heatmaps not for any fitted model but for a standard VGG16; default = False
    :type base_vgg: bool
    :param save: if set to True all generated heatmaps will be saved
    :type save: bool
    :return: None
    """
    # load model to calculate heatmaps on
    if base_vgg:
        model = None
        feature_model_output_layer = None
        type_of_model = 'vgg'
    else:
        print(
            os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'),
            " loading ...")
        model = load_model(
            os.path.join(STATIC_DIR, 'models', 'model_history_' + str(dataset_to_use) + str(suffix_path) + '.hdf5'))

        feature_model_output_layer = get_output_layer(model, type_of_model)

    # connect loaded model with data
    setup_model = train_eval_model(dataset_to_use, fit=False, type_of_model=type_of_model, suffix_path=suffix_path,
                                   model_for_feature_embedding=model,
                                   eval=False, loss=False, misclassified=False,
                                   feature_model_output_layer=feature_model_output_layer)

    # prepare data for further processing
    print("Preparing data ...")
    train_data = setup_model.dataset.data
    # x_test = setup_model.dataset.data_t
    # test_images = list(zip([setup_model.img_preprocess_for_prediction(file) for file in x_test],
    #                         [file.ground_truth_label for file in x_test],
    #                         [os.path.splitext(file.img_name)[0] for file in x_test]))
    # train_images = test_images
    train_images = list(zip([setup_model.img_preprocess_for_prediction(file) for file in train_data],
                            [file.ground_truth_label for file in train_data],
                            [os.path.splitext(file.img_name)[0] for file in train_data]))

    # Create model without trailing softmax for analyzer
    if base_vgg:
        model = VGG16(weights='imagenet', include_top=True)
        model_wo_softmax = innvestigate.model_wo_softmax(model)
    else:
        try:
            model.get_layer(name='dense')._name = 'dense_ori'
        except ValueError:
            pass  # when there is no layer named "dense" you don't need to rename it
        model_wo_softmax = innvestigate.model_wo_softmax(model)
    try:
        model_wo_softmax.get_layer(name='activation')._name = 'activation_ori'
    except ValueError:
        pass  # when there is no layer named "activation" you don't need to rename it

    # Set up analyzer
    print("Preparing analyzer ...")
    analyzer = innvestigate.create_analyzer(
        method,  # analysis method identifier as str eg "lrp.sequential_preset_a"
        model_wo_softmax,  # model without softmax output
        neuron_selection_mode="index",
        **parameters  # as dictionary eg {"epsilon": 0.1}
    )  # optional analysis parameters

    # if LRP heatmaps should be saved, first create paths to store images in
    # path will look like this: ./static/heatmaps/MultiCNN/dataset_name/label/imagename_heatmap.png
    if save:
        heatmap_directory = os.path.join(STATIC_DIR, 'heatmaps', train_data[0].fe.fe_model.name, dataset_to_use)
        print("Heatmaps will be saved to:\n", heatmap_directory)
        # create the folder for heatmaps if it is not created yet
        if not os.path.exists(heatmap_directory):
            os.makedirs(heatmap_directory)
            for output_neuron in range(len(setup_model.labelencoder.classes_)):
                output_neuron_label = output_neuron
                if not BINARY:
                    output_neuron_label = setup_model.labelencoder.inverse_transform([output_neuron])
                os.makedirs(os.path.join(heatmap_directory, output_neuron_label[0]))

    # for every image calculate heatmap
    tic = time.time()
    for image_nr, (x, y, img_name) in enumerate(train_images):
        if image_nr % 1000 == 0:
            print(image_nr, "*", len(setup_model.labelencoder.classes_), " LRP heatmaps created")

        # create heatmap for every label = output_neuron
        for output_neuron in range(len(setup_model.labelencoder.classes_)):
            output_neuron_label = output_neuron
            if not BINARY:
                output_neuron_label = setup_model.labelencoder.inverse_transform([output_neuron])
            heatmap_ = generate_LRP_heatmap(x, analyzer, output_neuron)
            # plt.imshow(heatmap_, interpolation='none')
            # plt.show()
            if save:
                save_path = os.path.join(heatmap_directory, output_neuron_label[0], img_name)
                plt.imsave(save_path + "_heatmap.png", heatmap_)

    toc = time.time()
    print("Heatmaps needed: ",
          "{}h {}min {}sec ".format(math.floor(((toc - tic) / (60 * 60))), math.floor(((toc - tic) % (60 * 60)) / 60),
                                    ((toc - tic) % 60)))


if __name__ == '__main__':
    from dataset import DataSet
    from feature_extractor import FeatureExtractor
    from modelsetup import ModelSetup

    # ===== Code to generate example heatmaps to find out which analyzer works best =====
    # EXAMPLE USAGES:
    # possible inputs for "methods":
    # methods_grayscale()   for grayscale data
    # methods_mnist()       for dataset mnist
    # methods_oct()         for datasets oct

    generate_method_comparison(dataset_to_use="mnist_1247", suffix_path="_cnn_seed3871", type_of_model="cnn",
                               methods=methods_mnist(), number_images=8)

    generate_method_comparison(dataset_to_use="oct_cc", suffix_path="_cnn_seed3871", type_of_model="cnn",
                               methods=methods_oct(), number_images=4)

    # generate_method_and_neuron_comparison(dataset_to_use="mnist_1247", suffix_path="_cnn_seed3871", type_of_model="cnn",
    #                                       methods=methods_mnist(), number_images=2)  # PresetA
    #
    # generate_method_and_neuron_comparison(dataset_to_use="oct_cc", suffix_path="_cnn_seed3871",
    #                                       type_of_model="cnn",
    #                                       methods=methods_oct(), number_images=3)  # Preset A flat

    # ===== End of try-out for analyzers =====

    # ===== Example usage for generating heatmaps for a whole dataset =====
    # if you want to change the color map of the heatmap got to "generate_LRP_heatmap()"
    # and change the parameter "cmap_type"

    # generate_LRP_heatmaps_for_dataset(dataset_to_use="oct_cc", suffix_path="_vgg4balanced",
    #                                   type_of_model="vgg",
    #                                   method="lrp.sequential_preset_a_flat", parameters={}, base_vgg=True)  # {"epsilon": 0.1}



