import random
import string

# Configurazione
NOME_FILE = "parole.txt"
TOTALE_PAROLE = 1_000_000
LUNGHEZZA = 8
CHUNK_SIZE = 50_000  # Scrittura in blocchi per ottimizzare le prestazioni

# Alfabeto di riferimento (usa string.ascii_letters per includere anche le maiuscole)
caratteri = string.ascii_lowercase

with open(NOME_FILE, "w", encoding="utf-8") as file:
    for _ in range(TOTALE_PAROLE // CHUNK_SIZE):
        blocco = [
            "".join(random.choices(caratteri, k=LUNGHEZZA)) + "\n"
            for _ in range(CHUNK_SIZE)
        ]
        file.writelines(blocco)

print(f"File '{NOME_FILE}' generato con successo con {TOTALE_PAROLE} parole.")