from __future__ import annotations
import unittest
from minsh.rolling_hashes import rolling_hash_string_pair_intersection

class TestSubstringIntersection(unittest.TestCase):
    def setUp(self):
        # Store for each pair of strings the substring lengths such that there is a shared substring of that length
        self.substrings_pairs = [
            # Common case including non-equal lengths
            ("abc", "abd", [1, 2]),
            # ("gattaca", "cgagttaga", [1, 2, 3]),
            # ("robots", "spinach", [1]),

            # # Zero length and no overlap
            # ("", "cgagttaga", []),
            # ("gattaca", "rohooobe", []),
            # ("", "", []),

            # # Equal strings
            # ("abcdefg", "abcdefg", [1, 2, 3, 4, 5, 6, 7]),
        ]

    def tearDown(self):
        pass

    def test(self):
        for string1, string2, shared_substring_lengths in self.substrings_pairs:
            assert max(shared_substring_lengths) <= min(len(string1), len(string2))

            shared_substring_lengths = set(shared_substring_lengths)
            for length in range(1, max(len(string1), len(string2))):
                should_find = length in shared_substring_lengths
                found = rolling_hash_string_pair_intersection(string1, string2, length)
                self.assertEqual(should_find, found)

if __name__ == "__main__":
    unittest.main()