import tempfile
import unittest
from pathlib import Path

from wisperauto.postprocess import MODE_CLEANED, MODE_RAW, MODE_REPORT, MODE_SMART, PostProcessor


class PostProcessorTest(unittest.TestCase):
    def _processor(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return PostProcessor(Path(tmpdir.name) / "voice_commands.json")

    def test_builds_raw_cleaned_smart_and_report_outputs(self):
        processor = self._processor()
        result = processor.build_outputs(
            "Bonjour virgule maitre point\n"
            "sauter ligne\n"
            "titre\n"
            "conclusions principales\n"
            "liste\n"
            "premiere demande point\n"
            "nouvelle puce\n"
            "seconde demande point\n"
            "fin de liste\n"
        )

        self.assertIn("sauter ligne", result.outputs[MODE_RAW])
        self.assertIn("Bonjour, maitre.", result.outputs[MODE_CLEANED])
        self.assertIn("# Conclusions principales", result.outputs[MODE_SMART])
        self.assertIn("- premiere demande.", result.outputs[MODE_SMART])
        self.assertIn("Compte rendu structure", result.outputs[MODE_REPORT])

    def test_keeps_ambiguous_command_as_content(self):
        processor = self._processor()
        result = processor.build_outputs(
            "il m'a demande de sauter ligne dans le document point\n"
        )

        self.assertIn(
            "sauter ligne dans le document.",
            result.outputs[MODE_SMART],
        )

    def test_applies_legal_number_rules_without_external_model(self):
        processor = self._processor()
        result = processor.build_outputs(
            "article mille deux cent quarante deux point\n"
            "montant vingt cinq euros point\n"
            "le douze janvier deux mille vingt six point\n"
            "numero de dossier a b c cent vingt trois point\n"
        )

        smart = result.outputs[MODE_SMART]
        self.assertIn("article 1242.", smart)
        self.assertIn("montant 25 €.", smart)
        self.assertIn("le 12 janvier 2026.", smart)
        self.assertIn("numero de dossier ABC-123.", smart)

    def test_correction_commands_modify_previous_text(self):
        processor = self._processor()
        result = processor.build_outputs(
            "premiere phrase point\n"
            "deuxieme phrase point\n"
            "supprime la derniere phrase\n"
            "troisieme phrase point\n"
            "supprime le dernier mot\n"
        )

        smart = result.outputs[MODE_SMART]
        self.assertIn("premiere phrase.", smart)
        self.assertNotIn("deuxieme phrase", smart)
        self.assertIn("troisieme", smart)
        self.assertNotIn("troisieme phrase.", smart)

    def test_handles_real_asr_line_break_variants(self):
        processor = self._processor()
        result = processor.build_outputs(
            "Mon cher confrere pointe sautée ligne je fais suite point de cette ligne veuillez point\n"
            "Pointe-sautes et ligne Votre bien devoue point\n"
        )

        smart = result.outputs[MODE_SMART]
        self.assertIn("Mon cher confrere.\nje fais suite.\nveuillez.", smart)
        self.assertIn("Votre bien devoue.", smart)
        self.assertNotIn("sautée ligne", smart)
        self.assertNotIn("Pointe-sautes", smart)

    def test_handles_dictated_parentheses_bullets_plural_and_email(self):
        processor = self._processor()
        result = processor.build_outputs(
            "Vous trouverez ci-dessous deux-points sautée ligne "
            "premier tiré De l'arrêt attaqué ouvrez une parenthèse court d'appel de Lyon fermez la parenthèse "
            "sautée ligne nouveau tiré les courriel au pluriel point "
            "adresse contact arobase christophebonan point fr point\n"
        )

        smart = result.outputs[MODE_SMART]
        self.assertIn("Vous trouverez ci-dessous:\n- De l'arrêt attaqué (cour d'appel de Lyon)", smart)
        self.assertIn("- les courriels.", smart)
        self.assertIn("contact@christophebonan.fr.", smart)
        self.assertNotIn("ouvrez une parenthèse", smart)
        self.assertNotIn("nouveau tiré", smart)

    def test_applies_safe_legal_corrections(self):
        processor = self._processor()
        result = processor.build_outputs(
            "vous trouverez 6 juin le rapport point\n"
            "extrait qu'abisse de la societe point\n"
            "procede verbal de restitution point\n"
            "compte Carpa point\n"
        )

        smart = result.outputs[MODE_SMART]
        self.assertIn("vous trouverez ci-joint le rapport.", smart)
        self.assertIn("extrait Kbis de la societe.", smart)
        self.assertIn("procès-verbal de restitution.", smart)
        self.assertIn("compte CARPA.", smart)

    def test_actions_are_isolated_between_build_outputs_calls(self):
        processor = self._processor()
        first = processor.build_outputs("Bonjour point\nsauter ligne\n")
        first_actions = list(first.actions)
        second = processor.build_outputs("Au revoir point\n")

        self.assertIn("Retour a la ligne.", first_actions)
        self.assertNotIn("Retour a la ligne.", second.actions)
        self.assertEqual(first.actions, first_actions)

    def test_replacement_command_handles_accents_in_existing_text(self):
        processor = self._processor()
        result = processor.build_outputs(
            "société Toré Films point\n"
            "remplace societe par entreprise\n"
            "première phrase point\n"
            "remplace premiere par seconde\n"
        )

        smart = result.outputs[MODE_SMART]
        self.assertIn("entreprise Toré Films.", smart)
        self.assertIn("seconde phrase.", smart)
        self.assertNotIn("société Toré Films.", smart)
        self.assertNotIn("première phrase.", smart)


if __name__ == "__main__":
    unittest.main()
