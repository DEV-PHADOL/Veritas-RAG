from ingestion.cleaner import clean_text


def test_clean_text():
    raw_text = """
        Reserve Bank of India


        Annual Report 2024-25


        Financial Stability
           and Banking.


        Capital adequacy ratio: 17.4%


        Total assets: ₹ 100,000 crore
    """

    cleaned = clean_text(raw_text)

    print("\nCleaned text:")
    print("-" * 50)
    print(cleaned)
    print("-" * 50)

    # Text should not be empty
    assert cleaned

    # Important content should remain
    assert "Reserve Bank of India" in cleaned
    assert "Annual Report 2024-25" in cleaned
    assert "Financial Stability" in cleaned
    assert "17.4%" in cleaned
    assert "₹ 100,000 crore" in cleaned

    # Excessive blank lines should be removed
    assert "\n\n\n" not in cleaned