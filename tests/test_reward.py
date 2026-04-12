"""Tests for src/evaluation/reward.py — the most testable pure-function module.

Coverage targets:
- normalize_text
- extract_numbers_with_context
- detect_unit_in_context
- normalize_number_with_units
- is_likely_year
- has_significant_text
- check_text_overlap
- extract_final_answer
- fuzzy_match_answer (the main entry point)
- score_answer
"""

import pytest

from src.evaluation.reward import (
    normalize_text,
    extract_numbers_with_context,
    detect_unit_in_context,
    normalize_number_with_units,
    is_likely_year,
    has_significant_text,
    check_text_overlap,
    extract_final_answer,
    fuzzy_match_answer,
    score_answer,
)


# ===========================================================================
# normalize_text
# ===========================================================================

class TestNormalizeText:
    def test_plain_ascii_unchanged(self):
        assert normalize_text("hello world") == "hello world"

    def test_unicode_minus_replaced_by_hyphen(self):
        result = normalize_text("\u2212 10")
        assert result == "- 10"

    def test_unicode_minus_literal_replaced(self):
        result = normalize_text("−5")
        assert result == "-5"

    def test_both_unicode_minus_forms(self):
        # The file replaces both '\u2212' and '−' (same codepoint, testing both branches)
        result = normalize_text("a\u2212b−c")
        assert result == "a-b-c"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="empty or None"):
            normalize_text("")

    def test_raises_on_none_like_falsy(self):
        with pytest.raises(ValueError):
            normalize_text(None)  # type: ignore[arg-type]

    def test_whitespace_preserved(self):
        result = normalize_text("  spaces  ")
        assert result == "  spaces  "


# ===========================================================================
# extract_numbers_with_context
# ===========================================================================

class TestExtractNumbersWithContext:
    def test_simple_integer(self):
        results = extract_numbers_with_context("543")
        assert len(results) == 1
        num, ctx, has_pct, is_neg = results[0]
        assert num == 543.0
        assert not has_pct
        assert not is_neg

    def test_decimal_number(self):
        results = extract_numbers_with_context("3.14")
        assert results[0][0] == pytest.approx(3.14)

    def test_percentage(self):
        results = extract_numbers_with_context("25%")
        num, ctx, has_pct, is_neg = results[0]
        assert num == 25.0
        assert has_pct is True

    def test_negative_number(self):
        results = extract_numbers_with_context("-7")
        num, ctx, has_pct, is_neg = results[0]
        assert num == -7.0
        assert is_neg is True

    def test_comma_separated_number(self):
        results = extract_numbers_with_context("1,000,000")
        assert results[0][0] == 1000000.0

    def test_multiple_numbers(self):
        results = extract_numbers_with_context("Revenue was 100 million and cost was 40 million")
        numbers = [r[0] for r in results]
        assert 100.0 in numbers
        assert 40.0 in numbers

    def test_unicode_minus_before_number(self):
        results = extract_numbers_with_context("−10")
        assert results[0][0] == -10.0

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            extract_numbers_with_context("")

    def test_context_window_captured(self):
        text = "The answer is 543 million dollars"
        results = extract_numbers_with_context(text)
        _, ctx, _, _ = results[0]
        assert "million" in ctx


# ===========================================================================
# detect_unit_in_context
# ===========================================================================

class TestDetectUnitInContext:
    @pytest.mark.parametrize("context,expected_unit,expected_mult", [
        ("543 million dollars", "million", 1e6),
        ("12 billion", "billion", 1e9),
        ("3 trillion", "trillion", 1e12),
        ("5 thousand employees", "thousand", 1e3),
        ("543 millions of people", "million", 1e6),
        ("2 billions of units", "billion", 1e9),
    ])
    def test_unit_detection(self, context, expected_unit, expected_mult):
        unit, mult = detect_unit_in_context(context)
        assert unit == expected_unit
        assert mult == pytest.approx(expected_mult)

    def test_no_unit_returns_none(self):
        unit, mult = detect_unit_in_context("the answer is 42")
        assert unit is None
        assert mult == pytest.approx(1.0)

    def test_empty_context(self):
        unit, mult = detect_unit_in_context("")
        assert unit is None
        assert mult == pytest.approx(1.0)

    def test_abbreviation_b_detected_as_billion(self):
        unit, mult = detect_unit_in_context("revenue is $5 b")
        assert unit == "billion"

    def test_abbreviation_m_detected_as_million(self):
        unit, mult = detect_unit_in_context("cost $3 m in total")
        assert unit == "million"

    def test_abbreviation_k_detected_as_thousand(self):
        unit, mult = detect_unit_in_context("salary is 80 k per year")
        assert unit == "thousand"


# ===========================================================================
# normalize_number_with_units
# ===========================================================================

class TestNormalizeNumberWithUnits:
    def test_million_context(self):
        base, unit = normalize_number_with_units(543, "543 million dollars")
        assert base == 543
        assert unit == "million"

    def test_no_unit_context(self):
        base, unit = normalize_number_with_units(42, "the answer is 42")
        assert base == 42
        assert unit is None

    def test_billion_context(self):
        base, unit = normalize_number_with_units(1.5, "1.5 billion assets")
        assert base == 1.5
        assert unit == "billion"

    def test_zero_value(self):
        base, unit = normalize_number_with_units(0, "zero value")
        assert base == 0


# ===========================================================================
# is_likely_year
# ===========================================================================

class TestIsLikelyYear:
    @pytest.mark.parametrize("year", [1900, 1977, 2000, 2023, 2100])
    def test_valid_years_detected(self, year):
        assert is_likely_year(float(year)) is True

    @pytest.mark.parametrize("not_year", [1899.0, 2101.0, 543.0, 1000000.0])
    def test_non_years_rejected(self, not_year):
        assert is_likely_year(not_year) is False

    def test_decimal_year_rejected(self):
        # 2023.5 is not an integer, so not a year
        assert is_likely_year(2023.5) is False

    def test_boundary_1900(self):
        assert is_likely_year(1900.0) is True

    def test_boundary_2100(self):
        assert is_likely_year(2100.0) is True


# ===========================================================================
# has_significant_text
# ===========================================================================

class TestHasSignificantText:
    def test_purely_numeric_no_significant_text(self):
        has_text, cleaned = has_significant_text("543 million")
        assert has_text is False

    def test_month_plus_year_has_significant_text(self):
        has_text, cleaned = has_significant_text("March 1977")
        assert has_text is True
        assert "march" in cleaned

    def test_just_number_no_significant_text(self):
        has_text, cleaned = has_significant_text("1234")
        assert has_text is False

    def test_empty_string(self):
        has_text, cleaned = has_significant_text("")
        assert has_text is False

    def test_date_with_month_name(self):
        has_text, cleaned = has_significant_text("April 15, 2020")
        assert has_text is True
        assert "april" in cleaned

    def test_percentage_alone_no_significant_text(self):
        has_text, cleaned = has_significant_text("25%")
        assert has_text is False

    def test_unit_words_removed(self):
        has_text, cleaned = has_significant_text("3 billions")
        assert has_text is False

    def test_single_letter_not_significant(self):
        # cleaned text shorter than 2 characters — not significant
        has_text, cleaned = has_significant_text("5 x")
        # "x" is 1 char, below threshold
        assert has_text is False


# ===========================================================================
# check_text_overlap
# ===========================================================================

class TestCheckTextOverlap:
    def test_matching_month_year(self):
        matches, rationale = check_text_overlap("March 1977", "March 1977")
        assert matches is True

    def test_different_months(self):
        matches, rationale = check_text_overlap("March 1977", "April 1977")
        assert matches is False

    def test_prediction_missing_month(self):
        matches, rationale = check_text_overlap("March 1977", "1977")
        assert matches is False

    def test_purely_numeric_gt_always_matches(self):
        # GT is purely numeric, so text check is skipped → True
        matches, rationale = check_text_overlap("543 million", "543 million dollars")
        assert matches is True

    def test_empty_ground_truth(self):
        matches, rationale = check_text_overlap("", "some prediction")
        assert matches is False

    def test_empty_prediction(self):
        matches, rationale = check_text_overlap("March 1977", "")
        assert matches is False


# ===========================================================================
# extract_final_answer
# ===========================================================================

class TestExtractFinalAnswer:
    def test_extracts_from_tags(self):
        text = "Some preamble\n<FINAL_ANSWER>42</FINAL_ANSWER>\nmore text"
        assert extract_final_answer(text) == "42"

    def test_returns_original_when_no_tags(self):
        text = "The answer is 42"
        assert extract_final_answer(text) == text

    def test_strips_whitespace_inside_tags(self):
        text = "<FINAL_ANSWER>  hello world  </FINAL_ANSWER>"
        assert extract_final_answer(text) == "hello world"

    def test_case_insensitive_tags(self):
        text = "<final_answer>result</final_answer>"
        assert extract_final_answer(text) == "result"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            extract_final_answer("")

    def test_raises_on_empty_tags(self):
        with pytest.raises(ValueError, match="empty"):
            extract_final_answer("<FINAL_ANSWER></FINAL_ANSWER>")

    def test_multiline_content_in_tags(self):
        text = "<FINAL_ANSWER>\nline1\nline2\n</FINAL_ANSWER>"
        assert "line1" in extract_final_answer(text)
        assert "line2" in extract_final_answer(text)


# ===========================================================================
# fuzzy_match_answer — core test battery
# ===========================================================================

class TestFuzzyMatchAnswerBasicNumeric:
    def test_exact_integer_match(self):
        is_correct, _ = fuzzy_match_answer("42", "42")
        assert is_correct is True

    def test_exact_decimal_match(self):
        is_correct, _ = fuzzy_match_answer("3.14", "3.14")
        assert is_correct is True

    def test_within_tolerance_5_percent(self):
        # 100 vs 104 → 4% diff, within default 5%
        is_correct, _ = fuzzy_match_answer("100", "The answer is 104", tolerance=0.05)
        assert is_correct is True

    def test_outside_tolerance(self):
        # 100 vs 200 → 100% diff
        is_correct, _ = fuzzy_match_answer("100", "200", tolerance=0.05)
        assert is_correct is False

    def test_negative_number_match(self):
        is_correct, _ = fuzzy_match_answer("-50", "-50")
        assert is_correct is True

    def test_negative_number_mismatch(self):
        is_correct, _ = fuzzy_match_answer("-50", "50")
        assert is_correct is False


class TestFuzzyMatchAnswerUnits:
    def test_million_base_match(self):
        # GT: "543 million" (base=543), Pred: "543 million" (base=543)
        is_correct, _ = fuzzy_match_answer("543 million", "543 million")
        assert is_correct is True

    def test_different_unit_scale_mismatch(self):
        # GT: "543 million" (base=543), Pred: "543000000" (base=543000000)
        # bases don't match → False
        is_correct, _ = fuzzy_match_answer("543 million", "543000000")
        assert is_correct is False

    def test_billion_match(self):
        is_correct, _ = fuzzy_match_answer("1.5 billion", "1.5 billion")
        assert is_correct is True


class TestFuzzyMatchAnswerYearFiltering:
    def test_year_gt_is_not_filtered(self):
        # When GT is a year, prediction years should not be filtered
        is_correct, _ = fuzzy_match_answer("1977", "The data was collected in 1977")
        assert is_correct is True

    def test_year_in_prediction_filtered_when_gt_not_year(self):
        # GT is 543 (not a year), pred has "2023" and "543"
        # "2023" should be filtered as a year; "543" should still match
        is_correct, _ = fuzzy_match_answer("543", "In 2023, the answer was 543")
        assert is_correct is True

    def test_year_only_in_prediction_no_match(self):
        # GT is 543, prediction only contains year 2023 → no match
        is_correct, _ = fuzzy_match_answer("543", "This was reported in 2023")
        assert is_correct is False


class TestFuzzyMatchAnswerHybrid:
    def test_month_year_exact_match(self):
        is_correct, _ = fuzzy_match_answer("March 1977", "March 1977")
        assert is_correct is True

    def test_month_year_wrong_month(self):
        is_correct, _ = fuzzy_match_answer("March 1977", "April 1977")
        assert is_correct is False

    def test_month_year_prediction_missing_month(self):
        is_correct, _ = fuzzy_match_answer("March 1977", "1977")
        assert is_correct is False


class TestFuzzyMatchAnswerMultiNumber:
    def test_multi_number_all_present(self):
        # GT has two numbers, both must appear in prediction
        is_correct, _ = fuzzy_match_answer("10 and 20", "values are 10 and 20")
        assert is_correct is True

    def test_multi_number_partial_match_fails(self):
        # GT has "10 and 20" but prediction only has "10"
        is_correct, _ = fuzzy_match_answer("10 and 20", "only 10 was found")
        assert is_correct is False


class TestFuzzyMatchAnswerTextComparison:
    def test_exact_text_match(self):
        is_correct, _ = fuzzy_match_answer("yes", "yes")
        assert is_correct is True

    def test_case_insensitive_text(self):
        is_correct, _ = fuzzy_match_answer("Yes", "yes")
        assert is_correct is True

    def test_gt_in_prediction_longer(self):
        is_correct, _ = fuzzy_match_answer("yes", "The answer is yes.")
        assert is_correct is True

    def test_text_mismatch(self):
        is_correct, _ = fuzzy_match_answer("yes", "no")
        assert is_correct is False

    def test_parenthetical_stripped_before_compare(self):
        # Parenthetical content is stripped from BOTH sides before comparing.
        # If both GT and prediction have matching parentheticals, they match.
        is_correct, _ = fuzzy_match_answer(
            "Federal Old-Age and Survivors Insurance (OASI) Trust Fund",
            "Federal Old-Age and Survivors Insurance (OASI) Trust Fund",
        )
        assert is_correct is True

    def test_parenthetical_differs_still_compares_cleaned(self):
        # After stripping parentheticals, "Trust Fund" appears in "Trust Fund here"
        is_correct, _ = fuzzy_match_answer(
            "Trust Fund (ABC)",
            "Trust Fund (XYZ)",
        )
        # After cleaning both become "trust fund"; they match
        assert is_correct is True

    def test_quoted_text_stripped(self):
        is_correct, _ = fuzzy_match_answer('"yes"', "yes")
        assert is_correct is True


class TestFuzzyMatchAnswerEdgeCases:
    def test_raises_on_empty_ground_truth(self):
        with pytest.raises(ValueError, match="Ground truth"):
            fuzzy_match_answer("", "some prediction")

    def test_raises_on_empty_prediction(self):
        with pytest.raises(ValueError, match="Predicted"):
            fuzzy_match_answer("42", "")

    def test_raises_on_invalid_tolerance_above_1(self):
        with pytest.raises(ValueError, match="Tolerance"):
            fuzzy_match_answer("42", "42", tolerance=1.5)

    def test_raises_on_invalid_tolerance_below_0(self):
        with pytest.raises(ValueError, match="Tolerance"):
            fuzzy_match_answer("42", "42", tolerance=-0.1)

    def test_tolerance_boundary_zero(self):
        # Exact match at tolerance=0
        is_correct, _ = fuzzy_match_answer("100", "100", tolerance=0.0)
        assert is_correct is True

    def test_tolerance_boundary_one(self):
        # Any numeric answer should match with tolerance=1.0 (100%)
        is_correct, _ = fuzzy_match_answer("100", "199", tolerance=1.0)
        assert is_correct is True

    def test_zero_value_gt_and_pred(self):
        is_correct, _ = fuzzy_match_answer("0", "The value is 0")
        assert is_correct is True

    def test_zero_gt_nonzero_pred(self):
        is_correct, _ = fuzzy_match_answer("0", "The value is 5")
        assert is_correct is False


class TestFuzzyMatchAnswerFinalAnswerTags:
    """Ensure fuzzy_match_answer works with pre-extracted final answers."""

    def test_numeric_in_extracted_answer(self):
        extracted = extract_final_answer("<FINAL_ANSWER>42</FINAL_ANSWER>")
        is_correct, _ = fuzzy_match_answer("42", extracted)
        assert is_correct is True

    def test_text_in_extracted_answer(self):
        extracted = extract_final_answer("<FINAL_ANSWER>yes</FINAL_ANSWER>")
        is_correct, _ = fuzzy_match_answer("yes", extracted)
        assert is_correct is True


# ===========================================================================
# score_answer
# ===========================================================================

class TestScoreAnswer:
    def test_correct_answer_scores_one(self):
        assert score_answer("42", "42") == 1.0

    def test_wrong_answer_scores_zero(self):
        assert score_answer("42", "99") == 0.0

    def test_close_answer_with_tolerance_scores_one(self):
        # 100 vs 104 → 4% → passes at 5% tolerance
        assert score_answer("100", "104", tolerance=0.05) == 1.0

    def test_text_match_scores_one(self):
        assert score_answer("yes", "yes") == 1.0

    def test_text_mismatch_scores_zero(self):
        assert score_answer("yes", "no") == 0.0

    @pytest.mark.parametrize("gt,pred,tol,expected", [
        ("100", "100", 0.0, 1.0),
        ("100", "105", 0.05, 1.0),   # within 5%
        ("100", "110", 0.05, 0.0),   # outside 5%
        ("March 1977", "March 1977", 0.0, 1.0),
        ("March 1977", "April 1977", 0.0, 0.0),
        ("yes", "Yes", 0.0, 1.0),
        ("no", "yes", 0.0, 0.0),
    ])
    def test_parametrized_score_answer(self, gt, pred, tol, expected):
        assert score_answer(gt, pred, tolerance=tol) == expected
