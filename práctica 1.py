import requests
from bs4 import BeautifulSoup

# URL de la página web que queremos extraer
url = 'https://example.com'

# Realizar la solicitud a la página web
response = requests.get(url)

# Crear objeto BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

print(response.text)

# Extraer datos específicos (por ejemplo, todos los # encabezados h2, h1, head, title, etc.)
headings = soup.find_all('h1')
i=1
for heading in headings:
    print("Result: ", i)
    print(heading.text)