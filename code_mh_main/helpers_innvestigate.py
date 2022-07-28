# All function in here are copied from the innvestigate git repository
# this also includes the TODOs
# https://github.com/albermax/innvestigate/

import innvestigate
import innvestigate.utils as iutils

import matplotlib.pyplot as plt


def postprocess(X):
    X = X.copy()
    X = iutils.postprocess_images(X)
    return X


def create_analyzers(data, methods, model_wo_softmax):
    """ Creates analyzers for all methods

    :param data: x_train data
    :param methods: vector of methods e.g. created with methods_grayscale()
    :return: analyzers
    """
    analyzers = []
    for method in methods:
        analyzer = innvestigate.create_analyzer(
            method[0],  # analysis method identifier
            model_wo_softmax,  # model without softmax output
            **method[1]
        )  # optional analysis parameters

        # Some analyzers require training.
        analyzer.fit(data, batch_size=256, verbose=1)
        analyzers.append(analyzer)
    return analyzers


def plot_image_grid(
    grid,
    row_labels_left,
    row_labels_right,
    col_labels,
    file_name=None,
    figsize=None,
    dpi=224,
):
    # TODO: reduce complexity

    n_rows = len(grid)
    n_cols = len(grid[0])
    if figsize is None:
        figsize = (n_cols, n_rows + 1)

    plt.clf()
    plt.rc("font", family="sans-serif")

    plt.figure(figsize=figsize)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows + 1, n_cols], loc=[r + 1, c])
            # No border around subplots
            for spine in ax.spines.values():
                spine.set_visible(False)
            # TODO controlled color mapping wrt all grid entries,
            # or individually. make input param
            if grid[r][c] is not None:
                ax.imshow(grid[r][c], interpolation="none")
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(
                        col_labels[c],
                        rotation=22.5,
                        horizontalalignment="left",
                        verticalalignment="bottom",
                    )

            # row labels
            if not c:
                if row_labels_left != []:
                    txt_left = [label + "\n" for label in row_labels_left[r]]
                    ax.set_ylabel(
                        "".join(txt_left),
                        rotation=0,
                        verticalalignment="center",
                        horizontalalignment="right",
                    )

            if c == n_cols - 1:
                if row_labels_right != []:
                    txt_right = [label + "\n" for label in row_labels_right[r]]
                    ax2 = ax.twinx()
                    # No border around subplots
                    for spine in ax2.spines.values():
                        spine.set_visible(False)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(
                        "".join(txt_right),
                        rotation=0,
                        verticalalignment="center",
                        horizontalalignment="left",
                    )

    if file_name is None:
        plt.show()
    else:
        print(f"Saving figure to {file_name}")
        plt.savefig(file_name, orientation="landscape", dpi=dpi, bbox_inches="tight")
        plt.show()
