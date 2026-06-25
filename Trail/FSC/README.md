Structured all files relatede to Parsing and chunking only 
rest all are still the same (scattered) 


# Document Parsing – How It Works

---

## 📄 PDF Parser

- Opens the PDF file
- Goes through each page one by one
- Splits every page into a **left half** and a **right half** to handle two-column layouts
- Reads text from both halves and merges them
- Cleans each line — removes junk characters, fixes broken spacing, collapses extra whitespace
- Throws away lines that are too short, too symbolic, or look like diagram noise
- Skips any line already seen before to avoid repetition
- Saves each page's cleaned text under its page number like `[Page 1]`, `[Page 2]`
- Bundles everything into one final output with full text + per-page breakdown

---

## 📊 PPTX Parser

- Opens the PowerPoint file
- Goes through each slide one by one
- For every slide, looks at each shape (text box, table, etc.)
- If a shape has text — cleans it and strips out any `"Slide X -"` label noise
- If a shape has a table — reads each row and joins the cells with ` | `
- Skips anything already seen before, both within a slide and across all slides
- Saves each slide's cleaned text and tables under its slide number like `[Slide 1]`, `[Slide 2]`
- Bundles everything into one final output with full text + per-slide breakdown

---

## 📝 DOCX Parser

- Opens the Word document
- Goes through each paragraph one by one
- Cleans the text — removes garbage characters, repeated letters, blank placeholders, and symbol-heavy lines
- Skips paragraphs that are empty or already seen before
- Detects if a paragraph is a **heading** and tags it as `[HEADING]` in the output
- After paragraphs, goes through every table in the document
- For each table row, joins the non-empty cells with ` | `
- Skips any table block already seen before
- Combines all paragraphs and tables into one final output with full text + structure breakdown