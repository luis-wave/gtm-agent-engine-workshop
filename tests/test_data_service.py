import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile


class UpdateProspectInfoTest(unittest.TestCase):
    def test_update_persists_to_source_and_profile(self):
        prospect_id = "LEAD-71001"
        record = data_service.PROSPECTS[prospect_id]
        original_tech_stack = list(record["tech_stack"])
        original_profile = data_service._PROFILES.pop(prospect_id, None)
        try:
            initial_profile = build_prospect_profile.invoke({"prospect_id": prospect_id})
            self.assertNotIn("Kafka", initial_profile["prospect_profile"]["tech_stack"])
            result = data_service.update_prospect_info(prospect_id, "Kafka")

            self.assertTrue(result["updated"])
            self.assertIn("Kafka", data_service.fetch_tech_stack(prospect_id))
            profile = build_prospect_profile.invoke({"prospect_id": prospect_id})
            self.assertIn("Kafka", profile["prospect_profile"]["tech_stack"])
        finally:
            record["tech_stack"] = original_tech_stack
            if original_profile is None:
                data_service._PROFILES.pop(prospect_id, None)
            else:
                data_service._PROFILES[prospect_id] = original_profile
