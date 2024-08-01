# Rosenbrock verification pipeline as a unit test.
import pytest
import random
import time
import keras
from smt.problems import Rosenbrock
from smt.sampling_methods import LHS

from airobas.blocks_hub.adv_block import CleverHansMultiIndexAdvBlock
from airobas.blocks_hub.decomon_block import DecomonBlock
from airobas.blocks_hub.marabou_block import MarabouBlock
from airobas.blocks_hub.gml_mip_block import GMLBrick
from airobas.verif_pipeline import ProblemContainer, BoundsDomainBoxParameterPerValueInterval, BoundsDomainBoxParameter, \
    StabilityProperty, full_verification_pipeline


def create_points(n_training: int = 20, n_test=200):
    ########### Initialization of the problem, construction of the training and validation points
    ndim = 2
    n_training = n_training
    # Define the function
    fun = Rosenbrock(ndim=ndim)
    # Construction of the DOE
    # in order to have the always same LHS points, random_state=1
    sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
    xt = sampling(n_training)
    # Compute the outputs
    yt = fun(xt)
    # Construction of the validation points
    n_test = n_test
    sampling = LHS(xlimits=fun.xlimits, criterion='ese', random_state=1)
    xtest = sampling(n_test)
    ytest = fun(xtest)
    return xt, yt, xtest, ytest, fun


def train_model(xt, yt, xtest, ytest, nb_epoch: int = 5000):
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    from sklearn.metrics import mean_squared_error
    # Generate some dummy data for training
    np.random.seed(42)
    # Define the neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=2, activation='relu'))#, kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu'))#, kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu'))#, kernel_regularizer=l2(0.01)))
    model.add(Dense(64, activation='relu'))#, kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    history = model.fit(xt, yt, epochs=nb_epoch, batch_size=32, validation_split=0.1, verbose=1)

    # Print the training history
    print("Training loss over epochs:")
    print(history.history['loss'])

    print("Validation loss over epochs:")
    print(history.history['val_loss'])
    y_pred = model.predict(xtest)
    test_loss = mean_squared_error(ytest, y_pred)
    print("Test Loss (Mean Squared Error):", test_loss)
    print()
    return model, y_pred


class RosenbrockContainer(ProblemContainer):
    @staticmethod
    def create_rosenbrock_container(model: keras.Model, abs_noise_input: float = 0.03,
                                    abs_noise_output: float = 10.0,
                                    rel_noise_output: float = 0.2,
                                    threshold_for_abs_noise: float = 200,
                                    use_different_zones_for_output: bool = True) -> "RosenbrockContainer":
        if use_different_zones_for_output:
            output_bound_domain_param = BoundsDomainBoxParameterPerValueInterval(
                    [(-float("inf"), -threshold_for_abs_noise,
                      BoundsDomainBoxParameter(rel_noise=rel_noise_output,
                                               use_relative=True)),
                     (-threshold_for_abs_noise, threshold_for_abs_noise,
                      BoundsDomainBoxParameter(abs_noise=abs_noise_output,
                                               use_relative=False)),
                     (threshold_for_abs_noise, float("inf"),
                      BoundsDomainBoxParameter(rel_noise=
                                               rel_noise_output,
                                               use_relative=True))
                     ])
        else:
            output_bound_domain_param = BoundsDomainBoxParameter(abs_noise=abs_noise_output,
                                                                 use_relative=False)
        stability_property = \
            StabilityProperty(
                input_bound_domain_param=BoundsDomainBoxParameter(abs_noise=abs_noise_input,
                                                                  use_relative=False),
                output_bound_domain_param=output_bound_domain_param)
        return RosenbrockContainer(tag_id="rosenbrock",
                                   model=model,
                                   stability_property=stability_property)


def main_script():
    xt, yt, xtest, ytest, fun = create_points(n_training=300, n_test=500)
    model, y_pred = train_model(xt, yt, xtest, ytest, nb_epoch=300)
    container = RosenbrockContainer.create_rosenbrock_container(model,
                                                                abs_noise_input=0.01,
                                                                abs_noise_output=10,
                                                                threshold_for_abs_noise=200,
                                                                use_different_zones_for_output=True)
    blocks = [(CleverHansMultiIndexAdvBlock, {"list_params_adv_block":
                                              [{"index_target": i,
                                                "attack_up": True,
                                                "fgs": True} for i in range(yt.shape[1])]})]
    # blocks += [(DecomonBlock, {}), (MarabouBlock, {"time_out": 100})]
    blocks += [(DecomonBlock, {}), (GMLBrick, {})]
    t1 = time.perf_counter()
    global_verif = full_verification_pipeline(problem=container,
                                              input_points=xtest,
                                              output_points=y_pred,  # or ytest if you target ground truth
                                              blocks_verifier=blocks,
                                              verbose=True)
    t2 = time.perf_counter()
    print(t2-t1, " seconds to run pipeline")
    print(xt.shape, yt.shape, xtest.shape, ytest.shape)


def test_rosenbrock_main():
    main_script()
