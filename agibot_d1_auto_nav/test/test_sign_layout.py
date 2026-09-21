from agibot_d1_auto_nav.sign_recognizer import SignRecognizer


def build_recognizer(digit_position):
    recognizer = SignRecognizer.__new__(SignRecognizer)
    recognizer.digit_position = digit_position
    return recognizer


def test_right_digit_layout_assigns_left_candidate_to_letter():
    left = object()
    right = object()

    digit, letter = build_recognizer('right')._assign_character_sides(left, right)

    assert digit is right
    assert letter is left


def test_left_digit_layout_restores_original_assignment():
    left = object()
    right = object()

    digit, letter = build_recognizer('left')._assign_character_sides(left, right)

    assert digit is left
    assert letter is right


def test_preloaded_character_classifier_is_reused():
    classifier = object()

    recognizer = SignRecognizer(classifier=classifier)

    assert recognizer.classifier is classifier
