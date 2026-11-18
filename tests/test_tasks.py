from filler_pipeline.tasks import make_dataset


def test_tasks_are_reproducible_and_answers_are_valid():
    first = make_dataset(20, 3)
    second = make_dataset(20, 3)
    assert first == second
    for task in first:
        assert task.answer.isdigit()
        assert task.filler
