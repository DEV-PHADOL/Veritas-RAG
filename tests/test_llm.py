from unittest.mock import Mock, patch

from generation.llm import (
    GeminiGenerator,
    GeminiServiceError
)


def test_gemini_retries_on_temporary_error():

    with patch("generation.llm.genai.Client") as mock_client:

        mock_instance = mock_client.return_value

        mock_instance.models.generate_content.side_effect = Exception(
            "503 UNAVAILABLE"
        )

        generator = GeminiGenerator(
            max_retries=2
        )

        with patch("generation.llm.time.sleep"):

            try:

                generator.generate("Test prompt")

                assert False, "Expected GeminiServiceError"

            except GeminiServiceError:

                pass

        assert (
            mock_instance.models.generate_content.call_count == 3
        )
        
        
def test_gemini_does_not_retry_permanent_error():

    with patch("generation.llm.genai.Client") as mock_client:

        mock_instance = mock_client.return_value

        mock_instance.models.generate_content.side_effect = Exception(
            "401 UNAUTHENTICATED"
        )

        generator = GeminiGenerator(
            max_retries=2
        )

        with patch("generation.llm.time.sleep") as mock_sleep:

            try:

                generator.generate("Test prompt")

                assert False, "Expected an exception"

            except Exception as error:

                assert "401" in str(error)

            mock_sleep.assert_not_called()

        assert (
            mock_instance.models.generate_content.call_count == 1
        )