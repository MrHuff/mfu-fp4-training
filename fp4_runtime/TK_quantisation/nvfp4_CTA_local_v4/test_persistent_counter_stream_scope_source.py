#!/usr/bin/env python3
"""CPU-only source contract for localCTA persistent-counter stream scoping."""

from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("tk_quantize.cu")


def _between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


class PersistentCounterStreamScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_counter_cache_is_per_host_thread_device_and_stream(self) -> None:
        accessor = _between(
            self.source,
            "struct LocalCTAPersistentCounter",
            "struct LocalCTAAtomicScratch",
        )
        self.assertIn(
            "static thread_local std::vector<LocalCTAPersistentCounter> entries",
            accessor,
        )
        self.assertIn(
            "at::cuda::getCurrentCUDAStream(device_index).stream()", accessor
        )
        self.assertIn("entry.device_index == device_index", accessor)
        self.assertIn("entry.stream_key == stream_key", accessor)
        self.assertNotIn("static std::vector<torch::Tensor> counters", accessor)

    def test_all_persistent_call_sites_use_the_scoped_accessor(self) -> None:
        # The accessor definition plus every persistent producer call must stay
        # routed through the stream-scoped cache rather than growing another
        # device-global counter cache.
        self.assertGreaterEqual(
            self.source.count("get_localcta_persistent_counter("), 14
        )
        self.assertEqual(
            self.source.count("static std::vector<torch::Tensor> counters"), 0
        )


if __name__ == "__main__":
    unittest.main()
