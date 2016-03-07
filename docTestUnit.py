from docAnalyzer import DocAnalyzer
import sys

docA = DocAnalyzer(sys.argv[1], sys.argv[2])
docA.parseDoc()