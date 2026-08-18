"""Guards on the evolver prompt and OpenRouter provider pinning.

Extracted from the former test_loop.py: the universal-prompt
benchmark-agnosticism rule and the provider hard-pin semantics.
"""


class TestUniversalPromptStaysBenchmarkAgnostic:
    """The evolver's system prompt + protocol must not name one benchmark's
    domain concepts — that content belongs in oracle/README.md.

    Standing rule; this test makes it enforceable rather than aspirational.
    """

    # "cell"/"cells" bare, not just "graded cell": the generalization section
    # said "no rules naming specific questions, cells, line numbers" — a
    # spreadsheet noun the narrower pattern missed (0724 over-prescription pass).
    _BENCHMARK_WORDS = (
        "xlsx", "spreadsheet", "openpyxl", "libreoffice", "answer_position",
        "cell", "workbook", "pandas", "docker verifier", "formula",
    )

    def test_system_prompt_names_no_benchmark_concept(self):
        from pathlib import Path

        import src.agent_profiles.skill_evolver as prof

        text = (Path(prof.__file__).parent / "prompt.md").read_text().lower()
        assert [w for w in self._BENCHMARK_WORDS if w in text] == []


class TestProviderHardPin:
    """`order` alone is a PREFERENCE; a run whose point IS the provider needs a pin.

    0722 rule: DSV4 must ride NATIVE DeepSeek (no :nitro). With only
    {"order": ["deepseek"]}, OpenRouter may quietly serve from another upstream
    when deepseek is busy -- and the numbers then belong to an endpoint nobody
    chose. Added 0726 when a DSV4F run was requested with "make sure to use
    deepseek as provider": the config expressed a wish, not a constraint.
    """

    def _extra_body(self, **opts):
        """Replay the executor's provider block on an options dict."""
        extra_body: dict = {}
        order = opts.get("provider_order")
        if order:
            extra_body["provider"] = {"order": order}
            allow = opts.get("provider_allow_fallbacks")
            if allow is not None:
                extra_body["provider"]["allow_fallbacks"] = bool(allow)
        return extra_body

    def test_order_alone_leaves_fallbacks_at_the_default(self):
        eb = self._extra_body(provider_order=["deepseek"])
        assert eb["provider"] == {"order": ["deepseek"]}
        assert "allow_fallbacks" not in eb["provider"]

    def test_false_hard_pins_the_provider(self):
        eb = self._extra_body(provider_order=["deepseek"],
                              provider_allow_fallbacks=False)
        assert eb["provider"] == {"order": ["deepseek"], "allow_fallbacks": False}

    def test_true_is_explicit_not_omitted(self):
        eb = self._extra_body(provider_order=["deepseek"],
                              provider_allow_fallbacks=True)
        assert eb["provider"]["allow_fallbacks"] is True

    def test_no_provider_block_without_an_order(self):
        assert self._extra_body(provider_allow_fallbacks=False) == {}

    def test_factory_threads_the_flag_into_agent_options(self):
        from src.agent_profiles.skill_evolver import make_direct_skill_evolver_options

        o = make_direct_skill_evolver_options(
            model="deepseek/deepseek-v4-flash", project_root="/tmp",
            budget_type="turns", budget_value=200,
            provider_order=["deepseek"], provider_allow_fallbacks=False,
        )
        assert o["provider_order"] == ["deepseek"]
        assert o["provider_allow_fallbacks"] is False
