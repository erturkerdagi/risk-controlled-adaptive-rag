from rarag.labels import contains_answer, normalize, score_answer


def test_normalize():
    assert normalize("The  Eiffel-Tower!") == "eiffel tower"


def test_word_boundary():
    assert contains_answer("Lillehammer", ["Lille"]) == 0.0
    assert contains_answer("It was Lille, in France", ["Lille"]) == 1.0
    assert contains_answer("Ashoka's father", ["Ashoka"]) == 1.0


def test_any_gold():
    assert contains_answer("Paris", ["London", "paris"]) == 1.0
    assert contains_answer("", ["Paris"]) == 0.0
    assert contains_answer("Paris", [""]) == 0.0


def test_score_answer():
    acc, em, f1 = score_answer("the Paris", ["Paris"])
    assert (acc, em, f1) == (1.0, 1.0, 1.0)
    acc, em, f1 = score_answer("Paris France", ["Paris"])
    assert acc == 1.0 and em == 0.0 and 0 < f1 < 1
