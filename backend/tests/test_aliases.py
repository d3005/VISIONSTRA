"""Label normalisation and class aliases ("Bike", "Pedestrian", "traffic_light")."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fields import normalise_name
from core.hazard_config import DEFAULT_CONFIG, config_from_dict
from core.risk import rank_detections, score_single, type_group_for


class TestNormaliseName(unittest.TestCase):

    def test_separators_case_and_spacing(self):
        for raw in ("Traffic_Light", "traffic-light", "  TRAFFIC   light ", "traffic light"):
            with self.subTest(raw=raw):
                self.assertEqual(normalise_name(raw), "traffic light")

    def test_blank_and_non_strings(self):
        for raw in (None, "", "   ", 5, ["car"]):
            with self.subTest(raw=raw):
                self.assertEqual(normalise_name(raw), "")

    def test_multi_word_coco_names_match_their_group(self):
        for raw in ("fire_hydrant", "Potted-Plant", "stop_sign", "parking-meter"):
            with self.subTest(raw=raw):
                self.assertEqual(type_group_for(raw)[0], "street_obstacle")


class TestDefaultAliases(unittest.TestCase):

    def test_everyday_labels_score_as_their_class(self):
        cases = {
            "Pedestrian": "person", "people": "person", "child": "person",
            "cycle": "bicycle", "Cyclist": "bicycle",
            "Bike": "vehicle", "motorbike": "vehicle", "Scooter": "vehicle",
            "two-wheeler": "vehicle", "auto_rickshaw": "vehicle", "Taxi": "vehicle",
            "lorry": "vehicle", "minibus": "vehicle",
            "buffalo": "animal", "cattle": "animal", "puppy": "animal",
        }
        for label, group in cases.items():
            with self.subTest(label=label):
                self.assertEqual(type_group_for(label)[0], group)

    def test_alias_scores_exactly_like_its_target(self):
        for alias, target in DEFAULT_CONFIG.scoring.class_aliases.items():
            with self.subTest(alias=alias):
                a = score_single(alias, 3.0, "Center", "approaching")
                b = score_single(target, 3.0, "Center", "approaching")
                self.assertEqual(a["risk_score"], b["risk_score"])
                self.assertEqual(a["priority"], b["priority"])
                self.assertEqual(a["category"], b["category"])

    def test_reason_keeps_the_label_the_detector_used(self):
        result = score_single("Bike", 2, "Center")
        self.assertTrue(result["reason"].startswith("Bike, 2 m"))
        self.assertIn("a vehicle", result["reason"])

    def test_unaliased_unknown_labels_stay_low(self):
        self.assertEqual(score_single("umbrella", 1, "Center")["category"], "other")

    def test_every_default_alias_targets_a_real_class(self):
        classes = {c for g in DEFAULT_CONFIG.scoring.type_groups.values() for c in g.classes}
        for alias, target in DEFAULT_CONFIG.scoring.class_aliases.items():
            with self.subTest(alias=alias):
                self.assertIn(target, classes)
                self.assertNotIn(alias, classes)
                self.assertEqual(alias, normalise_name(alias))


class TestTaskExampleWithEverydayLabels(unittest.TestCase):
    """The assignment's own example, with its own words, in every position."""

    def test_bike_is_primary_and_levels_match(self):
        for direction in ("Left", "Center", "Right", None):
            with self.subTest(direction=direction):
                ranked = rank_detections([
                    {"label": "Pedestrian", "distance": 8, "direction": direction},
                    {"label": "Car", "distance": 5, "direction": direction},
                    {"label": "Bike", "distance": 2, "direction": direction},
                ])
                order = [(d["label"], d["priority"]) for d in ranked]
                self.assertEqual(order[0][0], "Bike")
                self.assertEqual(order[0][1], "HIGH")
                self.assertEqual(dict(order)["Pedestrian"], "LOW")
                if direction is not None:
                    self.assertEqual(dict(order)["Car"], "MEDIUM")

    def test_before_this_change_bike_would_have_been_low(self):
        no_aliases = config_from_dict({"scoring": {"class_aliases": {
            alias: None for alias in DEFAULT_CONFIG.scoring.class_aliases}}}).scoring
        self.assertEqual(score_single("Bike", 2, "Center", config=no_aliases)["priority"], "LOW")
        self.assertEqual(score_single("Bike", 2, "Center")["priority"], "HIGH")


class TestAliasConfig(unittest.TestCase):

    def test_add_alias_is_normalised(self):
        cfg = config_from_dict({"scoring": {"class_aliases": {"E-Rickshaw": "CAR"}}})
        self.assertEqual(cfg.scoring.class_aliases["e rickshaw"], "car")
        self.assertEqual(type_group_for("e_rickshaw", cfg.scoring)[0], "vehicle")
        self.assertIn("pedestrian", cfg.scoring.class_aliases)   # defaults kept

    def test_override_and_remove_alias(self):
        cfg = config_from_dict({"scoring": {"class_aliases": {"bike": "bicycle", "auto": None}}})
        self.assertEqual(type_group_for("bike", cfg.scoring)[0], "bicycle")
        self.assertEqual(type_group_for("auto", cfg.scoring)[0], "other")

    def test_alias_to_class_in_new_group(self):
        cfg = config_from_dict({"scoring": {
            "type_groups": {"luggage": {"weight": 10, "distance_factor": 0.5,
                                        "description": "luggage", "classes": ["Suit_Case"]}},
            "class_aliases": {"trolley bag": "suit case"},
        }})
        self.assertEqual(type_group_for("Trolley-Bag", cfg.scoring)[0], "luggage")

    def test_invalid_aliases_are_rejected(self):
        bad = [
            {"class_aliases": {"hoverboard": "flying car"}},   # unknown target
            {"class_aliases": {"car": "truck"}},                # shadows a real class
            {"class_aliases": {"  ": "car"}},                   # blank alias
            {"class_aliases": ["bike", "motorcycle"]},          # wrong shape
        ]
        for overrides in bad:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    config_from_dict({"scoring": overrides})

    def test_defaults_are_not_modified_by_overrides(self):
        config_from_dict({"scoring": {"class_aliases": {"bike": "bicycle"}}})
        self.assertEqual(DEFAULT_CONFIG.scoring.class_aliases["bike"], "motorcycle")


if __name__ == "__main__":
    unittest.main()
