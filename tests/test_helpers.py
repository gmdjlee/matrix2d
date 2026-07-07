"""Tests for UI selection helpers (phase entries, gap-name parsing)."""

import numpy as np

from matrix2d.core.transform import TransformConfig
from matrix2d.ui import helpers


def _md(sample_no, time_s, temp_c, kind="TOP", path=None):
    return {
        "title": "T",
        "sample_no": sample_no,
        "time_s": time_s,
        "temp_c": temp_c,
        "kind": kind,
        "path": path or "{0}_{1}_{2}.dat".format(sample_no, time_s, temp_c),
    }


class TestPhaseEntries:
    def test_heating_and_cooling_split_at_peak(self):
        metas = [
            _md(1, 100, 200),   # heating
            _md(1, 200, 260),   # peak
            _md(1, 300, 200),   # cooling
        ]
        entries = helpers.phase_entries(metas)
        phases = [e["phase"] for e in entries]
        assert phases == ["H", "H", "C"]

    def test_peak_is_per_sample(self):
        metas = [
            _md(1, 100, 260), _md(1, 200, 200),   # sample 1 peaks at t=100
            _md(2, 100, 200), _md(2, 200, 260),   # sample 2 peaks at t=200
        ]
        entries = helpers.phase_entries(metas)
        by = {(e["sample_no"], e["time_s"]): e["phase"] for e in entries}
        assert by[(1, 100)] == "H"
        assert by[(1, 200)] == "C"
        assert by[(2, 100)] == "H"
        assert by[(2, 200)] == "H"

    def test_bad_dicts_skipped(self):
        metas = [_md(1, 100, 200), {"sample_no": "x"}]
        entries = helpers.phase_entries(metas)
        assert len(entries) == 1

    def test_partial_dict_with_valid_sample_no_skipped(self):
        # valid sample_no but missing the other SampleMeta fields must be
        # skipped per the docstring, not raise KeyError from meta_from_dict
        metas = [_md(1, 100, 200), {"sample_no": 2}]
        entries = helpers.phase_entries(metas)
        assert len(entries) == 1
        assert entries[0]["sample_no"] == 1

    def test_partial_dict_only_sample_skipped_entirely(self):
        # a sample whose every dict fails conversion produces no entries
        assert helpers.phase_entries([{"sample_no": 2}]) == []

    def test_empty(self):
        assert helpers.phase_entries([]) == []

    def test_explicit_phase_wins_over_peak_rule(self):
        # gap-named files carry phase directly; time_s=0 would otherwise
        # always classify as 'H' via the peak rule
        gap_c = dict(_md(1, 0, 250, kind="GAP"), phase="C", btm_no=12)
        entries = helpers.phase_entries([gap_c])
        assert entries[0]["phase"] == "C"

    def test_explicit_phase_does_not_skew_peak_of_others(self):
        # a phase-carrying dict for the same sample_no must not join the
        # peak-time computation of the normal measurement dicts
        metas = [
            _md(1, 100, 260),                                # peak at t=100
            _md(1, 200, 200),                                # cooling
            dict(_md(1, 0, 250, kind="GAP"), phase="H", btm_no=2),
        ]
        entries = helpers.phase_entries(metas)
        phases = [e["phase"] for e in entries]
        assert phases == ["H", "C", "H"]


class TestMetaDictRoundTrip:
    def test_gap_fields_roundtrip(self):
        from matrix2d.core.parser import parse_gap_filename
        meta = parse_gap_filename("TEST-H250_TOP1-BTM12.txt")
        back = helpers.meta_from_dict(helpers.meta_to_dict(meta))
        assert back == meta

    def test_old_dict_without_gap_fields(self):
        # dcc.Store dicts saved before btm_no/phase existed must still load
        meta = helpers.meta_from_dict(_md(1, 100, 200))
        assert meta.btm_no is None
        assert meta.phase is None

    def test_gap_label(self):
        from matrix2d.core.parser import parse_gap_filename
        meta = parse_gap_filename("TEST-H250_TOP1-BTM12.txt")
        assert helpers.meta_label(meta) == "GAP TOP1-BTM12 H250C"
        assert helpers.meta_label_from_dict(
            helpers.meta_to_dict(meta)) == "GAP TOP1-BTM12 H250C"


class TestPhaseTempKey:
    def test_encoding(self):
        assert helpers.phase_temp_key("H", 240) == "H240"
        assert helpers.phase_temp_key("C", 25) == "C25"


class TestSortPhaseTemps:
    def test_session_order(self):
        pairs = {("C", 200), ("H", 200), ("H", 260), ("C", 100)}
        assert helpers.sort_phase_temps(pairs) == [
            ("H", 200), ("H", 260), ("C", 200), ("C", 100)]


class TestParseGapName:
    def test_basic(self):
        p = helpers.parse_gap_name("TEST-C25_TOP3-BTM8.txt")
        assert p == {"top_no": 3, "btm_no": 8, "phase": "C", "temp_c": 25}

    def test_duplicate_suffix(self):
        p = helpers.parse_gap_name("TEST-H240_TOP1-BTM2_2.txt")
        assert p == {"top_no": 1, "btm_no": 2, "phase": "H", "temp_c": 240}

    def test_legacy_name(self):
        p = helpers.parse_gap_name("TOP1-BTM2_H240.txt")
        assert p == {"top_no": 1, "btm_no": 2, "phase": "H", "temp_c": 240}

    def test_legacy_duplicate_suffix(self):
        p = helpers.parse_gap_name("TOP12-BTM3_C85_2.txt")
        assert p == {"top_no": 12, "btm_no": 3, "phase": "C", "temp_c": 85}

    def test_no_match(self):
        assert helpers.parse_gap_name("random.txt") is None
        assert helpers.parse_gap_name("") is None


class TestBuildTransformConfig:
    def test_identity_returns_none(self):
        assert helpers.build_transform_config([], 0, None, None) is None
        assert helpers.build_transform_config(None, None, "", "") is None

    def test_flip_only(self):
        cfg = helpers.build_transform_config(["flip"], 0, None, None)
        assert cfg == TransformConfig(flip_lr=True, rot90_cw=0, zero_cell=None)

    def test_rotate_only_degrees_to_steps(self):
        cfg = helpers.build_transform_config([], 270, None, None)
        assert cfg == TransformConfig(flip_lr=False, rot90_cw=3, zero_cell=None)

    def test_zero_pair(self):
        cfg = helpers.build_transform_config(None, 0, 2, 3)
        assert cfg == TransformConfig(flip_lr=False, rot90_cw=0, zero_cell=(2, 3))

    def test_half_zero_pair_is_no_zero(self):
        # only row or only col set -> treated as "no zero cell"
        assert helpers.build_transform_config(None, 0, 2, None) is None
        assert helpers.build_transform_config(None, 0, None, 3) is None

    def test_string_inputs_coerced(self):
        cfg = helpers.build_transform_config(["flip"], "90", "1", "2")
        assert cfg == TransformConfig(flip_lr=True, rot90_cw=1, zero_cell=(1, 2))

    def test_garbage_inputs_ignored(self):
        assert helpers.build_transform_config([], "abc", "x", "y") is None


class TestTransformedMatrix:
    def _seed(self, arr, path):
        """Put an array straight into the matrix cache and return its meta."""
        helpers._MATRIX_CACHE[path] = arr
        return _md(1, 100, 200, kind="TOP", path=path)

    def test_flip_does_not_mutate_cache(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        md = self._seed(arr, "cache-flip.dat")
        out = helpers.transformed_matrix(md, TransformConfig(flip_lr=True))
        # cached raw matrix untouched
        assert np.array_equal(
            helpers._MATRIX_CACHE["cache-flip.dat"], [[1.0, 2.0], [3.0, 4.0]])
        # result is mirrored AND sign-inverted
        assert np.array_equal(out, [[-2.0, -1.0], [-4.0, -3.0]])

    def test_none_config_returns_copy_not_alias(self):
        arr = np.array([[1.0, 2.0]])
        md = self._seed(arr, "cache-copy.dat")
        out = helpers.transformed_matrix(md, None)
        assert out is not arr
        assert np.array_equal(out, arr)
