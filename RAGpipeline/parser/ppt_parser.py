# from pptx import Presentation
# from langchain_core.documents import Document


# def parse_ppt(file_path):

#     presentation = Presentation(file_path)

#     text = ""

#     for slide in presentation.slides:

#         for shape in slide.shapes:

#             if hasattr(shape, "text"):

#                 text += shape.text + "\n"

#     return [
#         Document(
#             page_content=text,
#             metadata={
#                 "source": file_path,
#                 "type": "pptx"
#             }
#         )
#     ]