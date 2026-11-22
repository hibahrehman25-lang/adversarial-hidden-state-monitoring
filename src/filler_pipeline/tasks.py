from filler_pipeline.tasks import make_dataset, make_matched_patching_pairs


def test_task_structure_and_filler_placement():
    ds = make_dataset(size=10, seed=42, filler_type="dots", filler_length=8, n_shots=3)
    assert len(ds) == 10
    for task in ds:
        assert 0 <= task.latent <= 9
        # Verify filler appears AFTER the final Question and BEFORE final Answer
        q_idx = task.prompt.rfind("Question:")
        f_idx = task.prompt.rfind("Filler:")
        a_idx = task.prompt.rfind("Answer:")
        assert q_idx < f_idx < a_idx


def test_matched_patching_pairs():
    pairs = make_matched_patching_pairs(n_pairs=5, seed=7, filler_type="count", filler_length=6)
    assert len(pairs) == 5
    for donor, target in pairs:
        assert donor.latent != target.latent
        assert donor.op_c == target.op_c
        assert donor.op_d == target.op_d
