from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from api.main import app


client = TestClient(app)


def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_invalid_query():

    response = client.post(
        "/ask",
        json={
            "query": "Hi"
        }
    )

    assert response.status_code == 422
    
def test_valid_query():

    mock_result = {
        "answer": "Cybersecurity awareness training helps employees recognize security threats.",
        "sources": [
            {
                "document_id": 11,
                "filename": "NIST_report.pdf",
                "chunk_id": 100,
                "page_number": 82,
                "content": "Cybersecurity awareness training information."
            }
        ],
        "retrieval_status": "relevant"
    }

    mock_pipeline = Mock()

    mock_pipeline.answer.return_value = mock_result

    with patch(
        "api.routes.get_pipeline",
        return_value=mock_pipeline
    ):

        response = client.post(
            "/ask",
            json={
                "query": "What is cybersecurity awareness training?"
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] != ""

    assert len(data["sources"]) == 1

    assert data["retrieval_status"] == "relevant"

    mock_pipeline.answer.assert_called_once()