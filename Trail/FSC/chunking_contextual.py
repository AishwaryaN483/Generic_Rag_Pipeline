import voyageai as vo

from Trail.FSC.src.Parsing.parsepdf import parse_pdf


print("paste the full path of the file:")
file_path = input().strip()
parsed_output = parse_pdf(file_path)

result = vo.Client().embed(texts=parsed_output, model="voyageai-4-large")
print(result)