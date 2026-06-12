import tempfile
import unittest
from pathlib import Path

from wisperauto.cancel import CancellationToken
from wisperauto.config import AppConfig
from wisperauto.errors import OperationCancelledError
from wisperauto.errors import PostProcessUnavailableError
from wisperauto.postprocess_llm import DirectLLMPostProcessor, SYSTEM_PROMPT, parse_smart_payload


class FakeDirectProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_smart_text(self, chunk, *, chunk_index, total_chunks):
        self.calls.append((chunk, chunk_index, total_chunks))
        return self.responses.pop(0)


class DirectLLMPostProcessorTest(unittest.TestCase):
    def test_system_prompt_distinguishes_formatting_from_business_commands(self):
        self.assertIn("Commandes de mise en forme", SYSTEM_PROMPT)
        self.assertIn("Consignes metier a conserver", SYSTEM_PROMPT)
        self.assertIn("vous scannez", SYSTEM_PROMPT)
        self.assertIn("vous faites un courriel", SYSTEM_PROMPT)
        self.assertIn("Pointe Sotéline", SYSTEM_PROMPT)
        self.assertIn("Gensotéline", SYSTEM_PROMPT)
        self.assertIn("ci-joint", SYSTEM_PROMPT)
        self.assertIn("point final puis retour a la ligne", SYSTEM_PROMPT)
        self.assertNotIn('veulent souvent dire ".\n".', SYSTEM_PROMPT)

    def test_applies_direct_smart_text_without_python_rules(self):
        config = AppConfig(home=Path("/tmp/wisperauto-test"))
        provider = FakeDirectProvider(
            [
                {
                    "smart_text": "Bonjour, maitre.\nVeuillez.",
                    "warnings": [],
                }
            ]
        )
        engine = DirectLLMPostProcessor(config, provider=provider)

        result = engine.apply("Bonjour virgule maitre point de sautée ligne veuillez")

        self.assertEqual(result.text, "Bonjour, maitre.\nVeuillez.\n")
        self.assertEqual(result.chunks, 1)
        self.assertEqual(result.warnings, [])
        self.assertEqual(provider.calls[0][1:], (0, 1))

    def test_does_not_apply_regex_fallback_when_provider_returns_literal_text(self):
        config = AppConfig(home=Path("/tmp/wisperauto-test"))
        provider = FakeDirectProvider(
            [
                {
                    "smart_text": "Bonjour virgule maitre point",
                    "warnings": [],
                }
            ]
        )
        engine = DirectLLMPostProcessor(config, provider=provider)

        result = engine.apply("Bonjour virgule maitre point")

        self.assertEqual(result.text, "Bonjour virgule maitre point\n")

    def test_reports_unavailable_when_model_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig(home=Path(tmpdir))
            engine = DirectLLMPostProcessor(config)

            with self.assertRaises(PostProcessUnavailableError) as raised:
                engine.apply("Bonjour point")

        self.assertIn("modele LLM local absent", str(raised.exception))

    def test_invalid_json_is_reported_as_postprocess_error(self):
        config = AppConfig(home=Path("/tmp/wisperauto-test"))
        engine = DirectLLMPostProcessor(config, provider=FakeDirectProvider(["pas du json"]))

        with self.assertRaises(PostProcessUnavailableError) as raised:
            engine.apply("Bonjour point")

        self.assertIn("reponse LLM invalide", str(raised.exception))

    def test_progress_callback_reports_chunks(self):
        config = AppConfig(home=Path("/tmp/wisperauto-test"))
        provider = FakeDirectProvider(
            [
                {"smart_text": "Premier.", "warnings": []},
                {"smart_text": "Deuxieme.", "warnings": []},
            ]
        )
        engine = DirectLLMPostProcessor(config, provider=provider, max_chunk_chars=10)
        events = []

        result = engine.apply("Premier.\nDeuxieme.", progress_callback=events.append)

        self.assertEqual(result.chunks, 2)
        self.assertIn("chunk_start", [event.stage for event in events])
        self.assertIn("chunk_done", [event.stage for event in events])
        self.assertEqual(events[-1].stage, "done")

    def test_cancel_between_chunks_stops_processing(self):
        config = AppConfig(home=Path("/tmp/wisperauto-test"))
        provider = FakeDirectProvider(
            [
                {"smart_text": "Premier.", "warnings": []},
                {"smart_text": "Deuxieme.", "warnings": []},
            ]
        )
        engine = DirectLLMPostProcessor(config, provider=provider, max_chunk_chars=10)
        token = CancellationToken()

        def cancel_after_first(event):
            if event.stage == "chunk_done" and event.completed_chunks == 1:
                token.cancel()

        with self.assertRaises(OperationCancelledError):
            engine.apply("Premier.\nDeuxieme.", progress_callback=cancel_after_first, cancel_token=token)

        self.assertEqual(len(provider.calls), 1)

    def test_parse_smart_payload_accepts_json_fence(self):
        text, warnings = parse_smart_payload(
            '```json\n{"smart_text":"Bonjour.\\n","warnings":["nom incertain"]}\n```'
        )

        self.assertEqual(text, "Bonjour.\n")
        self.assertEqual(warnings, ["nom incertain"])

    def test_parse_smart_payload_accepts_prefaced_json_fence(self):
        text, warnings = parse_smart_payload(
            'Voici la sortie JSON :\n```json\n{"smart_text":"Bonjour.","warnings":[]}\n```\nFin.'
        )

        self.assertEqual(text, "Bonjour.")
        self.assertEqual(warnings, [])

    def test_parse_smart_payload_accepts_json_object_with_trailing_text(self):
        text, warnings = parse_smart_payload(
            'Resultat:\n{"smart_text":"Bonjour.","warnings":["doute"]}\nMerci.'
        )

        self.assertEqual(text, "Bonjour.")
        self.assertEqual(warnings, ["doute"])

    def test_actions_do_not_accumulate_when_processor_is_reused(self):
        config = AppConfig(home=Path("/tmp/wisperauto-test"))
        provider = FakeDirectProvider(
            [
                {"smart_text": "Premier.", "warnings": []},
                {"smart_text": "Deuxieme.", "warnings": []},
            ]
        )
        engine = DirectLLMPostProcessor(config, provider=provider)

        first = engine.apply("Premier point")
        first_actions = list(engine.actions)
        second = engine.apply("Deuxieme point")

        self.assertEqual(first.text, "Premier.\n")
        self.assertEqual(second.text, "Deuxieme.\n")
        self.assertEqual(first_actions, ["Post-traitement LLM local : 1 segment(s) traite(s)."])
        self.assertEqual(engine.actions, ["Post-traitement LLM local : 1 segment(s) traite(s)."])


if __name__ == "__main__":
    unittest.main()
