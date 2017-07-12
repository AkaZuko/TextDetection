from docAnalyzer import DocAnalyzer
import sys

docA = DocAnalyzer(img_path=sys.argv[1], step_size_inp=sys.argv[2])
docA.parseDoc()
