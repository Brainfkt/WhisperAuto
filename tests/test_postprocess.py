import tempfile
import unittest
from pathlib import Path

from wisperauto.config import AppConfig
from wisperauto.errors import PostProcessUnavailableError
from wisperauto.postprocess import MODE_CLEANED, MODE_RAW, MODE_REPORT, MODE_SMART, PostProcessor


class FakeDirectProvider:
    def __init__(self, smart_text, warnings=None):
        self.smart_text = smart_text
        self.warnings = list(warnings or [])

    def generate_smart_text(self, _chunk, *, chunk_index, total_chunks):
        return {
            "smart_text": self.smart_text,
            "warnings": self.warnings,
        }


class PostProcessorTest(unittest.TestCase):
    def _processor(self, smart_text="Bonjour, maitre.\nVeuillez.", warnings=None):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        config = AppConfig(home=Path(tmpdir.name))
        return PostProcessor(
            config.commands_path,
            config=config,
            llm_provider=FakeDirectProvider(smart_text, warnings),
        )

    def test_builds_raw_smart_and_report_outputs_without_cleaned_mode(self):
        processor = self._processor("Bonjour, maitre.\nVeuillez.")
        result = processor.build_outputs("Bonjour virgule maitre point de sautée ligne veuillez\n")

        self.assertIn("Bonjour virgule maitre", result.outputs[MODE_RAW])
        self.assertEqual(result.outputs[MODE_SMART], "Bonjour, maitre.\nVeuillez.\n")
        self.assertIn("Compte rendu structure", result.outputs[MODE_REPORT])
        self.assertNotIn(MODE_CLEANED, result.outputs)

    def test_smart_output_is_only_provider_output(self):
        processor = self._processor("Bonjour virgule maitre point")
        result = processor.build_outputs("Bonjour virgule maitre point\n")

        self.assertEqual(result.outputs[MODE_SMART], "Bonjour virgule maitre point\n")
        self.assertNotIn("Bonjour, maitre.", result.outputs[MODE_SMART])

    def test_actions_include_llm_warnings(self):
        processor = self._processor("Texte final.", warnings=["nom propre incertain"])
        result = processor.build_outputs("Texte brut")

        self.assertTrue(any("Post-traitement LLM local" in action for action in result.actions))
        self.assertTrue(any("nom propre incertain" in action for action in result.actions))

    def test_missing_model_raises_clear_error_without_rules_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            processor = PostProcessor(config.commands_path, config=config)

            with self.assertRaises(PostProcessUnavailableError) as raised:
                processor.build_outputs("Bonjour virgule maitre point\n")

        self.assertIn("modele LLM local absent", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
