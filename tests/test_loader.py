from pathlib import Path

from ingestion.loader import load_pdf


def test_load_pdf():
    pdf_path = Path("data/raw/rbi_annual_report_2024_25.pdf")

    pages = load_pdf(str(pdf_path))

    assert len(pages) > 0
    assert pages[0]["page_number"] == 1

    non_empty_pages = [
        page for page in pages
        if page["text"]
    ]

    assert len(non_empty_pages) > 0

    print(f"\nTotal pages: {len(pages)}")
    print(f"Pages with text: {len(non_empty_pages)}")

    first_text_page = non_empty_pages[0]

    print(
        f"\nFirst page containing text: "
        f"{first_text_page['page_number']}"
    )

    print("\nText preview:")
    print(first_text_page["text"][:1000])