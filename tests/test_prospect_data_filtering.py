import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


FORBIDDEN_KEYS = {
    "billing_qualification",
    "tax_id",
    "date_of_birth",
    "card_on_file",
    "credit_check_ref",
}


def nested_keys(value):
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in nested_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in nested_keys(item)}
    return set()


class FakeScore:
    def model_dump(self):
        return {"score": 80}


class ProspectDataFilteringTest(unittest.TestCase):
    def test_prospect_tools_exclude_billing_data_from_scoring(self):
        data_service._PROFILES.clear()
        offering = next(iter(OFFERINGS.values()))

        scoring_llm = MagicMock()
        scoring_llm.invoke.return_value = FakeScore()
        with patch("gtm_agent.gtm_agent._scoring_llm", scoring_llm):
            for prospect_id in PROSPECTS:
                contact = get_prospect.invoke({"prospect_id": prospect_id})["prospect"]
                profile = build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
                score_prospect.invoke({"prospect_profile": profile, "offering": offering})

                self.assertEqual(set(contact), {"prospect_id", "name", "email", "disqualified"})
                self.assertEqual(
                    set(profile),
                    {
                        "prospect_id",
                        "name",
                        "email",
                        "annual_revenue",
                        "enrichment_source",
                        "engagement_history",
                        "account_details",
                        "tech_stack",
                    },
                )
                self.assertFalse(nested_keys(contact) & FORBIDDEN_KEYS)
                self.assertFalse(nested_keys(profile) & FORBIDDEN_KEYS)

            for call in scoring_llm.invoke.call_args_list:
                user_content = call.args[0][1]["content"]
                profile_json = user_content.split("\n\nProspect profile:\n", 1)[1]
                scoring_profile = json.loads(profile_json)
                self.assertEqual(
                    set(scoring_profile),
                    {"annual_revenue", "tech_stack", "account_details", "engagement_history"},
                )
                self.assertFalse(nested_keys(scoring_profile) & FORBIDDEN_KEYS)
