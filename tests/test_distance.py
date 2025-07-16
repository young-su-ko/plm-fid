from plm_fid.distance import calculate_frechet_distance
import numpy as np


class TestCalculateFrechetDistance:
    def test_identical_distributions(self):
        # Test case 1: Identical distributions
        mu_a = np.random.rand(1024)
        mu_b = mu_a.copy()

        sigma_a = np.eye(1024)
        sigma_b = sigma_a.copy()

        fid = calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)

        assert fid == 0.0

    def test_different_distributions(self):
        # Test case 2: Different distributions
        mu_a = np.ones(1024)
        mu_b = np.zeros(1024)

        sigma_a = np.eye(1024)
        sigma_b = np.eye(1024)

        fid = calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b)

        assert fid > 0.0
