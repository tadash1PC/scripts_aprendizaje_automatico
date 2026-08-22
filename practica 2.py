import requests

# URL de la API de GitHub
url = 'https://api.github.com/repos/python/cpython'

# Realizar la solicitud a la API
response = requests.get(url)
data = response.json()

# Mostrar información específica del repositorio
print(f"Repository: {data['name']}")
print(f"Description: {data['description']}")
print(f"Stars: {data['stargazers_count']}")
print(f"Forks: {data['forks_count']}")