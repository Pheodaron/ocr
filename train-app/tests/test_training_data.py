import pytest

from train_app.compute.data import response_span


class Tokenizer:
    eos_token_id = 99

    def decode(self, tokens, **_kwargs):
        return "".join({10: "0012", 11: ",30", 12: "</table>"}[token] for token in tokens)


def test_only_answer_and_eos_are_targets():
    assert response_span([1, 2, 3], [1, 2, 3, 10, 11, 99, 0], Tokenizer(), "0012,30", 20) == (3, 5)


@pytest.mark.parametrize(
    "prefix,full,answer,limit",
    [
        ([1, 2], [1, 5, 10, 99], "0012", 20),
        ([1, 2], [1, 2, 10], "0012", 20),
        ([1, 2], [1, 2, 10, 99], "0012", 3),
        ([1, 2], [1, 2, 10, 99], "0013", 20),
    ],
)
def test_bad_template_eos_truncation_and_changed_text_fail(prefix, full, answer, limit):
    with pytest.raises(ValueError):
        response_span(prefix, full, Tokenizer(), answer, limit)
