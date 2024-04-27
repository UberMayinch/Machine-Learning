#Phenomenology of Spirit- 32- 409
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (if not already downloaded)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfFileReader(file)
#         text = ''
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             text += page.extractText()
#     return text


def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming (you can also use lemmatization if needed)
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

def read_file_content(file_name):
    try:
        with open(file_name, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return f"Error: File '{file_name}' not found."
    except Exception as e:
        return f"An error occurred: {e}"

SOL=read_file_content('K/EoT.txt')
tokenized=preprocess_text(SOL)
print(tokenized)
