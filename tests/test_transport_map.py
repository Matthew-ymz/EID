import unittest

import numpy as np

from eid.estimators.transport_map import (
    AffineTransportMapDensityEstimator,
    _polynomial_design,
    _polynomial_exponents,
    estimate_mutual_information_transport_map,
    estimate_specific_mutual_information_transport_map,
    fit_affine_transport_map_density,
    fit_polynomial_triangular_transport_map_density,
    fit_quadratic_triangular_transport_map_density,
    multivariate_gaussian_logpdf,
    pairwise_effective_information_for_dynamics,
    standard_gaussian_logpdf,
)


class TransportMapDensityTests(unittest.TestCase):
    def test_memory_bounded_polynomial_design_matches_broadcast(self) -> None:
        rng = np.random.default_rng(7)
        for dimensions in (0, 1, 4):
            for degree in (1, 2, 3):
                values = rng.normal(size=(20, dimensions))
                mean, scale = values.mean(axis=0), values.std(axis=0)
                exponents = _polynomial_exponents(dimensions, degree)
                expected = np.prod(((values - mean) / scale)[:, None, :] ** exponents[None, :, :], axis=2)
                actual = _polynomial_design(values, exponents=exponents, mean=mean, scale=scale)
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_affine_transport_map_matches_gaussian_log_density(self) -> None:
        rng = np.random.default_rng(17)
        mean = np.array([0.8, -0.4], dtype=float)
        covariance = np.array([[1.6, 0.45], [0.45, 0.9]], dtype=float)
        train = rng.multivariate_normal(mean, covariance, size=6000)
        test = rng.multivariate_normal(mean, covariance, size=800)

        estimator = fit_affine_transport_map_density(train)

        self.assertIsInstance(estimator, AffineTransportMapDensityEstimator)
        log_hat = estimator.log_prob(test)
        log_true = multivariate_gaussian_logpdf(test, mean, covariance)
        rmse = float(np.sqrt(np.mean((log_hat - log_true) ** 2)))
        self.assertLess(rmse, 0.08)

    def test_pdf_matches_exponentiated_log_prob(self) -> None:
        rng = np.random.default_rng(23)
        samples = rng.normal(size=(200, 2))
        query = rng.normal(size=(20, 2))
        estimator = fit_affine_transport_map_density(samples)

        np.testing.assert_allclose(estimator.pdf(query), np.exp(estimator.log_prob(query)))

    def test_marginal_returns_working_density_estimator(self) -> None:
        rng = np.random.default_rng(31)
        samples = rng.normal(size=(300, 3))

        marginal = fit_affine_transport_map_density(samples).marginal([0])
        scores = marginal.log_prob(samples[:, [0]])

        self.assertEqual(marginal.dimension, 1)
        self.assertEqual(scores.shape, (samples.shape[0],))
        self.assertTrue(np.isfinite(scores).all())

    def test_quadratic_triangular_transport_map_matches_banana_density(self) -> None:
        rng = np.random.default_rng(41)
        beta = 0.42

        def banana_transform(base: np.ndarray) -> np.ndarray:
            warped = base.copy()
            warped[:, 1] = warped[:, 1] + beta * (warped[:, 0] ** 2 - 1.0)
            return warped

        def banana_inverse(samples: np.ndarray) -> np.ndarray:
            base = samples.copy()
            base[:, 1] = base[:, 1] - beta * (base[:, 0] ** 2 - 1.0)
            return base

        train = banana_transform(rng.normal(size=(5000, 2)))
        test = banana_transform(rng.normal(size=(1200, 2)))

        estimator = fit_quadratic_triangular_transport_map_density(train)
        log_hat = estimator.log_prob(test)
        log_true = standard_gaussian_logpdf(banana_inverse(test))
        rmse = float(np.sqrt(np.mean((log_hat - log_true) ** 2)))

        self.assertLess(rmse, 0.10)
        self.assertEqual(estimator.backend, "quadratic_triangular_transport_map")

    def test_polynomial_transport_map_specific_mi_averages_to_mutual_information(self) -> None:
        rng = np.random.default_rng(47)
        source = rng.uniform(-2.0, 2.0, size=(900, 1))
        target = np.sin(source) + 0.15 * rng.normal(size=(900, 1))

        estimator = fit_polynomial_triangular_transport_map_density(
            np.concatenate([target, source], axis=1),
            degree=3,
        )
        specific = estimate_specific_mutual_information_transport_map(
            source,
            target,
            target_anchors=target[::9],
            degree=3,
            conditional_samples=160,
            seed=5,
        )
        mutual = estimate_mutual_information_transport_map(source, target, degree=3)

        self.assertEqual(estimator.backend, "polynomial_triangular_transport_map_degree_3")
        self.assertEqual(specific["specific_mi"].shape, (100,))
        self.assertTrue(np.isfinite(specific["specific_mi"]).all())
        self.assertAlmostEqual(
            float(specific["specific_mi"].mean()),
            float(mutual["mi_hat"]),
            delta=0.25,
        )

    def test_uniform_sample_weights_match_unweighted_mutual_information(self) -> None:
        rng = np.random.default_rng(53)
        source = rng.normal(size=(700, 2))
        target = (source[:, [0]] - 0.4 * source[:, [1]]) + 0.3 * rng.normal(size=(700, 1))

        unweighted = estimate_mutual_information_transport_map(source, target, degree=3)
        weighted = estimate_mutual_information_transport_map(
            source,
            target,
            degree=3,
            sample_weight=np.ones(len(source)),
        )

        self.assertAlmostEqual(float(unweighted["mi_hat"]), float(weighted["mi_hat"]), places=12)
        self.assertAlmostEqual(float(weighted["effective_sample_size"]), len(source), places=10)

    def test_weighted_transport_map_tracks_weighted_first_moment(self) -> None:
        samples = np.array([[-2.0], [-1.0], [1.0], [3.0]], dtype=float)
        weights = np.array([1.0, 1.0, 2.0, 6.0], dtype=float)

        estimator = fit_polynomial_triangular_transport_map_density(
            samples,
            degree=2,
            sample_weight=weights,
        )

        expected_mean = float(np.average(samples[:, 0], weights=weights))
        self.assertAlmostEqual(float(estimator.coefficients[0][0]), expected_mean, places=12)

    def test_polynomial_exponents_degree_one_scales_linearly_in_dimension(self) -> None:
        exponents = _polynomial_exponents(64, 1)

        self.assertEqual(exponents.shape, (65, 64))
        self.assertTrue(np.all(exponents.sum(axis=1) <= 1))

    def test_pairwise_effective_information_for_dynamics_returns_indexed_matrix(self) -> None:
        def dynamics(inputs: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    inputs[:, 0] + 0.05 * inputs[:, 1],
                    inputs[:, 2],
                ]
            )

        summary = pairwise_effective_information_for_dynamics(
            dynamics,
            input_indices=[0, 1, 2],
            output_indices=[0, 1],
            input_dim=3,
            box_size=4.0,
            sample_count=3500,
            seed=19,
        )

        pairwise = summary["pairwise_ei"]

        self.assertEqual(pairwise.shape, (3, 2))
        self.assertEqual(summary["input_indices"], [0, 1, 2])
        self.assertEqual(summary["output_indices"], [0, 1])
        self.assertGreater(float(pairwise[0, 0]), float(pairwise[1, 0]))
        self.assertGreater(float(pairwise[2, 1]), 1.0)
        self.assertLess(float(pairwise[0, 1]), 0.05)

    def test_pairwise_effective_information_for_dynamics_supports_scalar_output(self) -> None:
        def dynamics(inputs: np.ndarray) -> np.ndarray:
            return 2.0 * inputs[:, 1]

        summary = pairwise_effective_information_for_dynamics(
            dynamics,
            input_indices=[0, 1],
            output_indices=[0],
            input_dim=2,
            sample_count=2500,
            seed=23,
        )

        pairwise = summary["pairwise_ei"]

        self.assertEqual(pairwise.shape, (2, 1))
        self.assertLess(float(pairwise[0, 0]), 0.05)
        self.assertGreater(float(pairwise[1, 0]), 1.0)

    def test_pairwise_effective_information_for_dynamics_accepts_scalar_indices(self) -> None:
        def dynamics(inputs: np.ndarray) -> np.ndarray:
            return inputs[:, 0]

        summary = pairwise_effective_information_for_dynamics(
            dynamics,
            input_indices=0,
            output_indices=0,
            sample_count=1500,
            seed=29,
        )

        self.assertEqual(summary["pairwise_ei"].shape, (1, 1))
        self.assertEqual(summary["input_indices"], [0])
        self.assertEqual(summary["output_indices"], [0])


if __name__ == "__main__":
    unittest.main()
