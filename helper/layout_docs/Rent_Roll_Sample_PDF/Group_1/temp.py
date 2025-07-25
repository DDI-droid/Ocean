import PyPDF2

def rotate_pdf_90_clockwise(file_path):
    # Open the existing PDF
    with open(file_path, "rb") as infile:
        reader = PyPDF2.PdfReader(infile)
        writer = PyPDF2.PdfWriter()

        # Assuming it's a single-page PDF
        page = reader.pages[0]
        page.rotate(90)  # Rotate 90 degrees clockwise

        writer.add_page(page)

        # Overwrite the original PDF
        with open(file_path, "wb") as outfile:
            writer.write(outfile)

# Example usage
rotate_pdf_90_clockwise("/home/arys/Documents/Rent_Roll_Sample_1/Rent_Roll_Sample_PDF/Group_1/WFRBS 2014-C24_Bend River Promenade_20231231.pdf")

