
from __future__ import annotations

import tempfile
from typing import Any, List, Mapping, Optional, Tuple, Type, cast

import gpflow
import numpy.testing as npt
import pytest
import tensorflow as tf
from _pytest.mark import ParameterSet

from tests.util.misc import random_seed
from tests.util.models.keras.models import MCDropConnect
from trieste.acquisition import (
    GIBBON,
    AcquisitionFunctionClass,
    AugmentedExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    Fantasizer,
    GreedyContinuousThompsonSampling,
    LocalPenalization,
    MinValueEntropySearch,
    MultipleOptimismNegativeLowerConfidenceBound,
    ParallelContinuousThompsonSampling,
)
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousGreedy,
    AsynchronousOptimization,
    AsynchronousRuleState,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    TrustRegion,
)
from trieste.acquisition.sampler import ThompsonSamplerFromTrajectory
from trieste.bayesian_optimizer import BayesianOptimizer, TrainableProbabilisticModelType
from trieste.logging import tensorboard_writer
from trieste.models import TrainableProbabilisticModel, TrajectoryFunctionClass
from trieste.models.gpflow import (
    GaussianProcessRegression,
    GPflowPredictor,
    SparseVariational,
    VariationalGaussianProcess,
    build_gpr,
    build_svgp,
)
from trieste.models.gpflux import DeepGaussianProcess, build_vanilla_deep_gp
from trieste.models.keras import DeepEnsemble, build_vanilla_keras_ensemble, MCDropout, build_vanilla_keras_mcdropout
from trieste.models.keras.architectures import DropoutNetwork, DropConnectNetwork
from trieste.models.optimizer import BatchOptimizer, KerasOptimizer
from trieste.objectives import (
    BRANIN_MINIMIZERS,
    BRANIN_SEARCH_SPACE,
    SCALED_BRANIN_MINIMUM,
    SIMPLE_QUADRATIC_MINIMIZER,
    SIMPLE_QUADRATIC_MINIMUM,
    SIMPLE_QUADRATIC_SEARCH_SPACE,
    scaled_branin,
    simple_quadratic,
)
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box, SearchSpace
from trieste.types import State, TensorType

@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(5, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(5, DiscreteThompsonSampling(500, 1), id="DiscreteThompsonSampling")
    ],
)
def test_bayesian_optimizer_with_mcdropout_finds_minima_of_simple_quadratic(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace, MCDropout]
) -> None:
    _test_optimizer_finds_minimum(MCDropout, num_steps, acquisition_rule)

@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        # pytest.param(90, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(30, DiscreteThompsonSampling(500, 3), id="DiscreteThompsonSampling")
    ],
)
def test_bayesian_optimizer_with_mcdropout_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, MCDropout],
) -> None:
    _test_optimizer_finds_minimum(
        MCDropout,
        num_steps,
        acquisition_rule,
        optimize_branin=True
    )


@random_seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        pytest.param(5, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(5, DiscreteThompsonSampling(500, 1), id="DiscreteThompsonSampling")
    ],
)
def test_bayesian_optimizer_with_mcdropconnect_finds_minima_of_simple_quadratic(
    num_steps: int, acquisition_rule: AcquisitionRule[TensorType, SearchSpace, MCDropConnect]
) -> None:
    _test_optimizer_finds_minimum(MCDropConnect, num_steps, acquisition_rule)

@random_seed
# @pytest.mark.slow
@pytest.mark.parametrize(
    "num_steps, acquisition_rule",
    [
        # pytest.param(90, EfficientGlobalOptimization(), id="EfficientGlobalOptimization"),
        pytest.param(30, DiscreteThompsonSampling(500, 3), id="DiscreteThompsonSampling")
    ],
)
def test_bayesian_optimizer_with_mcdropconnect_finds_minima_of_scaled_branin(
    num_steps: int,
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, MCDropConnect],
) -> None:
    _test_optimizer_finds_minimum(
        MCDropConnect,
        num_steps,
        acquisition_rule,
        optimize_branin=True
    )


def _test_optimizer_finds_minimum(
    model_type: Type[TrainableProbabilisticModelType],
    num_steps: Optional[int],
    acquisition_rule: AcquisitionRule[TensorType, SearchSpace, TrainableProbabilisticModelType]
    | AcquisitionRule[
        State[TensorType, AsynchronousRuleState | TrustRegion.State],
        Box,
        TrainableProbabilisticModelType,
    ],
    optimize_branin: bool = False,
    model_args: Optional[Mapping[str, Any]] = None,
) -> None:
    model_args = model_args or {}
    track_state = True

    if optimize_branin:
        search_space = BRANIN_SEARCH_SPACE
        minimizers = BRANIN_MINIMIZERS
        minima = SCALED_BRANIN_MINIMUM
        rtol_level = 0.005
        num_initial_query_points = 5
    else:
        search_space = SIMPLE_QUADRATIC_SEARCH_SPACE
        minimizers = SIMPLE_QUADRATIC_MINIMIZER
        minima = SIMPLE_QUADRATIC_MINIMUM
        rtol_level = 0.05
        num_initial_query_points = 10

    if model_type in [SparseVariational, DeepGaussianProcess, DeepEnsemble]:
        num_initial_query_points = 20

    initial_query_points = search_space.sample(num_initial_query_points)
    observer = mk_observer(scaled_branin if optimize_branin else simple_quadratic)
    initial_data = observer(initial_query_points)

    model: TrainableProbabilisticModel  # (really TPMType, but that's too complicated for mypy)

    if model_type is GaussianProcessRegression:
        if "LocalPenalization" in acquisition_rule.__repr__():
            likelihood_variance = 1e-3
        else:
            likelihood_variance = 1e-5
        gpr = build_gpr(initial_data, search_space, likelihood_variance=likelihood_variance)
        model = GaussianProcessRegression(gpr, **model_args)

    elif model_type is VariationalGaussianProcess:
        empirical_variance = tf.math.reduce_variance(initial_data.observations)
        kernel = gpflow.kernels.Matern52(variance=empirical_variance, lengthscales=[0.2, 0.2])
        likelihood = gpflow.likelihoods.Gaussian(1e-3)
        vgp = gpflow.models.VGP(initial_data.astuple(), kernel, likelihood)
        gpflow.utilities.set_trainable(vgp.likelihood, False)
        model = VariationalGaussianProcess(vgp, **model_args)

    elif model_type is SparseVariational:
        svgp = build_svgp(initial_data, search_space)
        model = SparseVariational(svgp, **model_args)

    elif model_type is DeepGaussianProcess:
        track_state = False
        dgp = build_vanilla_deep_gp(initial_data, search_space)
        model = DeepGaussianProcess(dgp, **model_args)

    elif model_type is DeepEnsemble:
        track_state = False

        keras_ensemble = build_vanilla_keras_ensemble(initial_data, 5, 3, 25)
        fit_args = {
            "batch_size": 20,
            "epochs": 1000,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss", patience=25, restore_best_weights=True
                )
            ],
            "verbose": 0,
        }
        de_optimizer = KerasOptimizer(tf.keras.optimizers.Adam(0.001), fit_args)
        model = DeepEnsemble(keras_ensemble, de_optimizer, **model_args)
    
    elif model_type is MCDropout or model_type is MCDropConnect:
        track_state = False
        
        if model_type is MCDropConnect:
            dropout_network = build_vanilla_keras_mcdropout(initial_data, rate=0.2, dropout_network=DropConnectNetwork)
        else:
            dropout_network = build_vanilla_keras_mcdropout(initial_data, rate=0.03, dropout_network=DropoutNetwork)
        
        fit_args = {
            "batch_size": 32,
            "epochs": 1000,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(monitor="loss", patience=80), 
                tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.3, patience=15)
                ],
            "verbose": 0
        }

        mc_optimizer = KerasOptimizer(tf.keras.optimizers.Adam(learning_rate=0.001), fit_args)

        model = MCDropout(dropout_network, mc_optimizer, num_passes=200, **model_args)

    else:
        raise ValueError(f"Unsupported model_type '{model_type}'")

    with tempfile.TemporaryDirectory() as tmpdirname:
        summary_writer = tf.summary.create_file_writer(tmpdirname)
        with tensorboard_writer(summary_writer):

            dataset = (
                BayesianOptimizer(observer, search_space)
                .optimize(
                    num_steps or 2,
                    initial_data,
                    cast(TrainableProbabilisticModelType, model),
                    acquisition_rule,
                    track_state=track_state,
                )
                .try_get_final_dataset()
            )

            arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
            best_y = dataset.observations[arg_min_idx]
            best_x = dataset.query_points[arg_min_idx]

            if num_steps is None:
                # this test is just being run to check for crashes, not performance
                pass
            else:
                minimizer_err = tf.abs((best_x - minimizers) / minimizers)
                # these accuracies are the current best for the given number of optimization
                # steps, which makes this is a regression test
                assert tf.reduce_any(tf.reduce_all(minimizer_err < 0.05, axis=-1), axis=0)
                npt.assert_allclose(best_y, minima, rtol=rtol_level)

            # check that acquisition functions defined as classes aren't retraced unnecessarily
            # They should be retraced once for the optimzier's starting grid, L-BFGS, and logging.
            if isinstance(acquisition_rule, EfficientGlobalOptimization):
                acq_function = acquisition_rule._acquisition_function
                if isinstance(acq_function, (AcquisitionFunctionClass, TrajectoryFunctionClass)):
                    assert acq_function.__call__._get_tracing_count() == 3  # type: ignore
