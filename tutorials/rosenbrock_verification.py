# Part of this tutorial uses SMT library :
# and particularly about this tutorial :
# https://github.com/SMTorg/smt/blob/master/tutorial/SMT_Tutorial.ipynb
import os
import random
import time

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from airobas.blocks_hub.adv_block import CleverHansMultiIndexAdvBlock
from airobas.blocks_hub.decomon_block import DecomonBlock
from airobas.blocks_hub.marabou_block import MarabouBlock
from airobas.verif_pipeline import (
    BoundsDomainBoxParameter,
    BoundsDomainBoxParameterPerValueInterval,
    ProblemContainer,
    StabilityProperty,
    StatusVerif,
    compute_bounds,
    full_verification_pipeline,
)
from decomon.models import clone
from mpl_toolkits.mplot3d import Axes3D

# from matplotlib.cm import get_cmap
from smt.problems import NdimRobotArm, Rosenbrock, Sphere
from smt.sampling_methods import LHS
from smt.utils.misc import compute_rms_error

this_folder = os.path.dirname(os.path.abspath(__file__))
image_dump_folder = os.path.join(this_folder, "rosenbrock_images/")
if not os.path.exists(image_dump_folder):
    os.makedirs(image_dump_folder)


def create_points(n_training: int = 20, n_test=200):
    ########### Initialization of the problem, construction of the training and validation points
    ndim = 2
    n_training = n_training
    # Define the function
    fun = Rosenbrock(ndim=ndim)
    # Construction of the DOE
    # in order to have the always same LHS points, random_state=1
    sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
    xt = sampling(n_training)
    # Compute the outputs
    yt = fun(xt)
    # Construction of the validation points
    n_test = n_test
    sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
    xtest = sampling(n_test)
    ytest = fun(xtest)
    return xt, yt, xtest, ytest, fun


def create_points_grid(grid_size: int = 51, fraction_training: float = 0.2):
    fun = Rosenbrock(ndim=2)
    x = np.linspace(-2, 2, grid_size)
    res = []
    points = []
    for x0 in x:
        for x1 in x:
            res.append(fun(np.array([[x0, x1]])))
            points.append([x0, x1])
    random_indexes = set(random.sample(range(len(points)), k=int(fraction_training * len(points))))
    xt = np.array([points[i] for i in random_indexes])
    yt = np.array([res[i] for i in random_indexes])
    xtest = np.array([points[i] for i in range(len(points)) if i not in random_indexes])
    ytest = np.array([res[i][0] for i in range(len(points)) if i not in random_indexes])
    return xt, yt, xtest, ytest, fun


def plot_3d(xt, yt, xtest, ytest, fun, name_figure="Rosenbrock function"):
    x = np.linspace(-2, 2, 51)
    res = []
    for x0 in x:
        for x1 in x:
            res.append(fun(np.array([[x0, x1]])))
    res = np.array(res)
    res = res.reshape((51, 51)).T
    X, Y = np.meshgrid(x, x)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, res, cmap=matplotlib.colormaps["viridis"], linewidth=0, antialiased=False, alpha=0.5)
    if xt is not None:
        ax.scatter(xt[:, 0], xt[:, 1], yt, zdir="z", marker="x", c="b", s=200, label="Training point")
    if xtest is not None:
        ax.scatter(xtest[:, 0], xtest[:, 1], ytest, zdir="z", marker=".", c="k", s=200, label="Validation point")
    plt.title(name_figure)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()


def train_model(xt, yt, xtest, ytest, nb_epoch: int = 5000):
    import numpy as np
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.regularizers import l2
    from sklearn.metrics import mean_squared_error

    # Generate some dummy data for training
    np.random.seed(42)
    # Define the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation="relu"))  # , kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation="relu"))  # , kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation="relu"))  # , kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation="relu"))  # , kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation="linear"))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train the model
    history = model.fit(xt, yt, epochs=nb_epoch, batch_size=32, validation_split=0.1, verbose=1)

    # Print the training history
    print("Training loss over epochs:")
    print(history.history["loss"])

    print("Validation loss over epochs:")
    print(history.history["val_loss"])
    y_pred = model.predict(xtest)
    test_loss = mean_squared_error(ytest, y_pred)
    print("Test Loss (Mean Squared Error):", test_loss)
    print()
    return model, y_pred


class RosenbrockContainer(ProblemContainer):
    @staticmethod
    def create_rosenbrock_container(
        model: keras.Model,
        abs_noise_input: float = 0.03,
        abs_noise_output: float = 10.0,
        rel_noise_output: float = 0.2,
        threshold_for_abs_noise: float = 200,
        use_different_zones_for_output: bool = True,
    ) -> "RosenbrockContainer":
        if use_different_zones_for_output:
            output_bound_domain_param = BoundsDomainBoxParameterPerValueInterval(
                [
                    (
                        -float("inf"),
                        -threshold_for_abs_noise,
                        BoundsDomainBoxParameter(rel_noise=rel_noise_output, use_relative=True),
                    ),
                    (
                        -threshold_for_abs_noise,
                        threshold_for_abs_noise,
                        BoundsDomainBoxParameter(abs_noise=abs_noise_output, use_relative=False),
                    ),
                    (
                        threshold_for_abs_noise,
                        float("inf"),
                        BoundsDomainBoxParameter(rel_noise=rel_noise_output, use_relative=True),
                    ),
                ]
            )
        else:
            output_bound_domain_param = BoundsDomainBoxParameter(abs_noise=abs_noise_output, use_relative=False)
        stability_property = StabilityProperty(
            input_bound_domain_param=BoundsDomainBoxParameter(abs_noise=abs_noise_input, use_relative=False),
            output_bound_domain_param=output_bound_domain_param,
        )
        return RosenbrockContainer(tag_id="rosenbrock", model=model, stability_property=stability_property)


def main_script():
    xt, yt, xtest, ytest, fun = create_points(n_training=300, n_test=500)
    model, y_pred = train_model(xt, yt, xtest, ytest, nb_epoch=300)
    decomon_computation(model)
    plt.show()
    container = RosenbrockContainer.create_rosenbrock_container(
        model,
        abs_noise_input=0.01,
        abs_noise_output=10,
        threshold_for_abs_noise=200,
        use_different_zones_for_output=True,
    )
    blocks = [
        (
            CleverHansMultiIndexAdvBlock,
            {
                "list_params_adv_block": [
                    {"index_target": i, "attack_up": True, "fgs": True} for i in range(yt.shape[1])
                ]
            },
        )
    ]
    blocks += [(DecomonBlock, {}), (MarabouBlock, {"time_out": 100})]
    t1 = time.perf_counter()
    global_verif = full_verification_pipeline(
        problem=container,
        input_points=xtest,
        output_points=y_pred,  # or ytest if you target ground truth
        blocks_verifier=blocks,
        verbose=True,
    )
    t2 = time.perf_counter()
    print(t2 - t1, " seconds to run pipeline")
    print(xt.shape, yt.shape, xtest.shape, ytest.shape)
    # To visualize the DOE points
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(xt[:, 0], xt[:, 1], marker="x", c="b", s=200, label="Training points")
    plt.scatter(xtest[:, 0], xtest[:, 1], marker=".", c="k", s=200, label="Validation points")
    plt.title("DOE")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plot_3d(xt, yt, xtest, ytest, fun)
    indexes = np.nonzero(global_verif.status == StatusVerif.VIOLATED)
    cnt_x = np.array([x for x in global_verif.inputs if x is not None])
    cnt_y = np.array(
        [global_verif.outputs[i] for i in range(len(global_verif.outputs)) if global_verif.outputs[i] is not None]
    )
    print("expected", [y_pred[i] for i in range(len(global_verif.outputs)) if global_verif.outputs[i] is not None])
    print("cnt", cnt_y)
    plot_3d(None, None, cnt_x, cnt_y, fun)
    plt.show()


def main_script_grid():
    xt, yt, xtest, ytest, fun = create_points_grid(grid_size=51, fraction_training=0.3)
    xt, yt, xtest, ytest, fun = create_points(n_training=100, n_test=1000)
    model, y_pred = train_model(xt, yt, xtest, ytest, nb_epoch=1500)
    decomon_computation(model)
    container = RosenbrockContainer.create_rosenbrock_container(
        model,
        abs_noise_input=0.03,
        abs_noise_output=20,
        rel_noise_output=0.25,
        threshold_for_abs_noise=200,
        use_different_zones_for_output=True,
    )
    blocks = [
        (
            CleverHansMultiIndexAdvBlock,
            {
                "list_params_adv_block": [
                    {"index_target": i, "attack_up": True, "fgs": True} for i in range(yt.shape[1])
                ]
            },
        )
    ]
    blocks += [(DecomonBlock, {}), (MarabouBlock, {"time_out": 100})]
    t1 = time.perf_counter()

    global_verif = full_verification_pipeline(
        problem=container,
        input_points=xtest,
        output_points=y_pred,  # or ytest if you target ground truth
        blocks_verifier=blocks,
        verbose=True,
    )
    t2 = time.perf_counter()
    print(t2 - t1, " seconds to run pipeline")
    print(xt.shape, yt.shape, xtest.shape, ytest.shape)
    # To visualize the DOE points
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(xt[:, 0], xt[:, 1], marker="x", c="b", s=200, label="Training points")
    plt.scatter(xtest[:, 0], xtest[:, 1], marker=".", c="k", s=200, label="Validation points")
    plt.title("DOE")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plot_3d(xt, yt, xtest, ytest, fun)
    indexes = np.nonzero(global_verif.status == StatusVerif.VIOLATED)
    cnt_x = np.array([x for x in global_verif.inputs if x is not None])
    cnt_y = np.array(
        [global_verif.outputs[i] for i in range(len(global_verif.outputs)) if global_verif.outputs[i] is not None]
    )
    y_exp_min, y_exp_max = compute_bounds(container.stability_property, y_pred, is_input=False)
    print(
        "expected",
        [
            (y_exp_min[i], y_pred[i], y_exp_max[i], global_verif.outputs[i])
            for i in range(len(global_verif.outputs))
            if global_verif.outputs[i] is not None
        ],
    )
    decomon_model = clone(model)
    for index_counter_example in range(cnt_x.shape[0]):
        original_point = xtest[indexes[0][index_counter_example]]
        # x_val = cnt_x[index_counter_example, 0]
        x_val = original_point[0]
        y_val = original_point[1]
        expected_value = fun(np.array([original_point]))
        found_value = cnt_y[index_counter_example]
        y = np.linspace(max(-2, y_val - 1), min(2, y_val + 1), 100)
        vals = np.array([[x_val, yi] for yi in y])
        x_min_, x_max_ = compute_bounds(container.stability_property, vals, is_input=True)
        box = np.concatenate([x_min_[:, None], x_max_[:, None]], 1)
        y_up_, y_low_ = decomon_model.predict(box)
        ground_truth = fun(vals)
        output_property = model.predict(vals)
        y_min_, y_max_ = compute_bounds(container.stability_property, output_property, is_input=False)
        fig, ax = plt.subplots(1)
        ax.plot(y, output_property, color="purple", label="model prediction")
        ax.plot(y, ground_truth, color="green", marker="*", label="ground truth")

        ax.plot(y, y_low_, color="blue", label="lower bound decomon")
        ax.plot(y, y_up_, color="red", label="upper bound decomon")

        ax.plot(y, y_min_, color="blue", linestyle="--", label="lower bound property")
        ax.plot(y, y_max_, color="red", linestyle="--", label="upper bound property")
        ax.scatter([y_val], [found_value], marker="x", s=500)  # , label="cnt example")
        ax.legend()
        fig.savefig(os.path.join(image_dump_folder, f"cnt_example_{index_counter_example}.png"))
        plt.close(fig)

    plot_3d(None, None, cnt_x, cnt_y, fun)
    plt.show()


def decomon_computation(model: keras.Model):
    fun = Rosenbrock(ndim=2)
    x = np.linspace(-2, 2, 51)
    res = []
    points = []
    for x0 in x:
        for x1 in x:
            res.append(fun(np.array([[x0, x1]])))
            points.append([x0, x1])
    points = np.array(points)
    lower = points - 0.1
    upper = points + 0.1
    decomon_model = clone(model)
    box = np.concatenate([lower[:, None], upper[:, None]], 1)
    t2 = time.perf_counter()
    y_up, y_low = decomon_model.predict(box)
    res = np.array(res)
    prediction = model.predict(points).reshape((51, 51)).T
    res = res.reshape((51, 51)).T
    y_up = y_up.reshape((51, 51)).T
    y_low = y_low.reshape((51, 51)).T
    X, Y = np.meshgrid(x, x)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        X, Y, res, cmap=matplotlib.colormaps["Blues"], linewidth=0, antialiased=False, alpha=0.5, label="ground_truth"
    )
    surf1 = ax.plot_surface(
        X,
        Y,
        prediction,
        cmap=matplotlib.colormaps["Greens"],
        linewidth=0,
        antialiased=False,
        alpha=0.5,
        label="surrogate",
    )
    surf2 = ax.plot_surface(
        X,
        Y,
        y_up,
        cmap=matplotlib.colormaps["Reds"],
        linewidth=0,
        antialiased=False,
        alpha=0.5,
        label="upper bound decomon",
    )
    surf3 = ax.plot_surface(
        X,
        Y,
        y_low,
        cmap=matplotlib.colormaps["Oranges"],
        linewidth=0,
        antialiased=False,
        alpha=0.5,
        label="lower bound decomon",
    )
    plt.title("Rosenbrock")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    for j in range(Y.shape[0]):
        fig, ax = plt.subplots(1)
        ax.plot(X[j, :], res[j, :], color="blue", label="ground_truth")
        ax.plot(X[j, :], prediction[j, :], color="green", label="surrogate")
        ax.plot(X[j, :], y_up[j, :], color="red", label="upper bound decomon")
        ax.plot(X[j, :], y_low[j, :], color="orange", label="lower bound decomon")
        ax.legend()
        slice = str(round(Y[j, 0], 2))
        #        ax.set_title(f"Slice x2 = {Y[j,0]}")
        #        fig.savefig(os.path.join(image_dump_folder, f"slice_x2eq{Y[j,0]}.png"))
        ax.set_title(f"Slice x2 = " + slice)
        ax.set_xlabel("x1")
        fig.savefig(os.path.join(image_dump_folder, f"slice_x2eq" + slice + ".png"))
        plt.close(fig)

    for j in range(X.shape[0]):
        fig, ax = plt.subplots(1)
        ax.plot(Y[:, j], res[:, j], color="blue", label="ground_truth")
        ax.plot(Y[:, j], prediction[:, j], color="green", label="surrogate")
        ax.plot(Y[:, j], y_up[:, j], color="red", label="upper bound decomon")
        ax.plot(Y[:, j], y_low[:, j], color="orange", label="lower bound decomon")
        ax.legend()
        slice = str(round(X[0, j], 2))
        ax.set_title(f"Slice x1 = " + slice)
        ax.set_xlabel("x2")
        fig.savefig(os.path.join(image_dump_folder, f"slice_x1eq" + slice + ".png"))
        plt.close(fig)


if __name__ == "__main__":
    main_script_grid()
