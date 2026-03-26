import numpy as np

from src.data.dataset import load_real_dataset, make_splits


def test_load_data_structure_shapes_and_ranges():
    ds = load_real_dataset(
        dataset_root="dataset",
        dataset_name="dataStructure",
        n_theory=58,
        n_experiment=10,
        expected_k=19,
        score_scale=10.0,
        missing_value=-1.0,
        q_normalize=True,
        q_zero_row_strategy="unknown_skill",
    )

    assert ds.n_students == 96
    assert ds.n_theory == 58
    assert ds.n_experiment == 10
    assert ds.n_skills == 19

    assert ds.r_theory.shape == (96, 58)
    assert ds.r_experiment.shape == (96, 10)
    assert ds.q_theory.shape == (58, 19)
    assert ds.q_experiment.shape == (10, 19)

    obs_theory = ds.r_theory[ds.mask_theory]
    obs_exp = ds.r_experiment[ds.mask_experiment]
    assert obs_theory.min() >= 0.0
    assert obs_theory.max() <= 1.0
    assert obs_exp.min() >= 0.0
    assert obs_exp.max() <= 1.0

    assert np.allclose(ds.q_theory.sum(axis=1), 1.0)
    assert np.allclose(ds.q_experiment.sum(axis=1), 1.0)


def test_load_network_security_unknown_skill_and_normalized_q():
    ds = load_real_dataset(
        dataset_root="dataset",
        dataset_name="networkSecurity",
        n_theory=10,
        n_experiment=8,
        expected_k=7,
        score_scale=10.0,
        missing_value=-1.0,
        q_normalize=True,
        q_zero_row_strategy="unknown_skill",
    )

    assert ds.n_students == 194
    assert ds.n_theory == 10
    assert ds.n_experiment == 8
    assert ds.n_skills == 7

    assert np.allclose(ds.combined_q().sum(axis=1), 1.0)

    unknown_skill_col = ds.combined_q()[:, -1]
    assert unknown_skill_col.sum() > 0.0


def test_make_splits_covers_observed_entries_combined_mode():
    ds = load_real_dataset(
        dataset_root="dataset",
        dataset_name="dataStructure",
        n_theory=58,
        n_experiment=10,
        expected_k=19,
        score_scale=10.0,
        missing_value=-1.0,
        q_normalize=True,
        q_zero_row_strategy="unknown_skill",
    )

    spl = make_splits(ds, train_ratio=0.8, val_ratio=0.0, seed=123, split_mode="combined")

    total_obs = int(ds.combined_mask().sum())
    covered = int(
        spl.train_theory.sum()
        + spl.train_experiment.sum()
        + spl.val_theory.sum()
        + spl.val_experiment.sum()
        + spl.test_theory.sum()
        + spl.test_experiment.sum()
    )
    assert covered == total_obs

    overlap = (
        (spl.train_theory & spl.test_theory).sum()
        + (spl.train_theory & spl.val_theory).sum()
        + (spl.val_theory & spl.test_theory).sum()
        + (spl.train_experiment & spl.test_experiment).sum()
        + (spl.train_experiment & spl.val_experiment).sum()
        + (spl.val_experiment & spl.test_experiment).sum()
    )
    assert int(overlap) == 0
