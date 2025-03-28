# analyse_sujet.py

# Ce script analyse automatiquement le sujet principal d’un texte en français
# Il utilise le NLP avec spaCy et des méthodes de machine learning

import spacy
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Télécharger les stopwords en français
nltk.download('stopwords')
from nltk.corpus import stopwords


# Charger le modèle spaCy en français
nlp = spacy.load("fr_core_news_md")

# Fonction pour charger un texte depuis un fichier
def charger_texte(fichier):
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            texte = f.read()
        return texte
    except Exception as e:
        print("Erreur lors de la lecture du fichier :", e)
        return ""

# Fonction de nettoyage du texte
def nettoyer_texte(texte):
    # Conversion en minuscules
    texte = texte.lower()

    # Suppression de la ponctuation et des chiffres
    texte = texte.translate(str.maketrans("", "", string.punctuation + "0123456789"))

    # Tokenisation et lemmatisation avec spaCy
    doc = nlp(texte)
    mots_utiles = [token.lemma_ for token in doc 
                   if token.is_alpha and token.lemma_ not in stopwords.words("french")]

    return " ".join(mots_utiles)

# Fonction pour afficher les mots les plus fréquents
def afficher_frequence_mots(texte_nettoye, n=20):
    mots = texte_nettoye.split()
    compteur = Counter(mots)
    mots_communs = compteur.most_common(n)
    
    print("\nLes mots les plus fréquents :")
    for mot, freq in mots_communs:
        print(f"{mot} : {freq}")

    return compteur

# Fonction pour générer un WordCloud
def generer_wordcloud(compteur):
    nuage = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(compteur)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(nuage, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nuage de mots")
    plt.show()



# Fonction pour extraire les termes les plus importants avec TF-IDF
def extraire_termes_tfidf(texte_nettoye, top_n=10):
    # Le vectoriseur TF-IDF a besoin d'une liste de documents
    documents = [texte_nettoye]

    # Initialisation du vectoriseur
    vecteur = TfidfVectorizer()
    tfidf_matrix = vecteur.fit_transform(documents)

    # Récupération des scores
    scores = tfidf_matrix.toarray()[0]
    termes = vecteur.get_feature_names_out()

    # Association terme <-> score
    resultat = list(zip(termes, scores))

    # Tri par score décroissant
    resultat = sorted(resultat, key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} termes par TF-IDF :")
    for mot, score in resultat[:top_n]:
        print(f"{mot} : {score:.4f}")

    return resultat[:top_n]

# Fonction pour extraire les entités nommées d’un texte
def extraire_entites_nommees(texte):
    doc = nlp(texte)
    entites = [(ent.text, ent.label_) for ent in doc.ents]

    print("\nEntités nommées trouvées :")
    for texte, type_entite in entites:
        print(f"{texte} -> {type_entite}")

    return entites


# Dictionnaire de thèmes et leurs mots-clés
themes_mots_cles = {
    "Technologie": ["intelligence", "artificielle", "algorithme", "robot", "numérique"],
    "Politique": ["élection", "gouvernement", "loi", "ministre", "président"],
    "Économie": ["marché", "finance", "investissement", "banque", "chômage"],
    "Santé": ["hôpital", "médecin", "maladie", "vaccin", "santé"],
    "Sport": ["match", "football", "compétition", "joueur", "tournoi"]
}

# Fonction de classification par mots-clés
def classer_texte_par_theme(texte_nettoye, themes):
    mots = set(texte_nettoye.split())
    scores = {}

    for theme, mots_cles in themes.items():
        score = len(mots.intersection(mots_cles))
        scores[theme] = score

    theme_choisi = max(scores, key=scores.get)
    if scores[theme_choisi] == 0:
        theme_choisi = "Inconnu"

    print(f"\nThème détecté (par mots-clés) : {theme_choisi}")
    return theme_choisi

# Fonction pour charger plusieurs textes d’un dossier
def charger_corpus(dossier):
    textes = []
    noms_fichiers = []
    for nom_fichier in os.listdir(dossier):
        if nom_fichier.endswith(".txt"):
            chemin_fichier = os.path.join(dossier, nom_fichier)
            texte = charger_texte(chemin_fichier)
            texte_nettoye = nettoyer_texte(texte)
            textes.append(texte_nettoye)
            noms_fichiers.append(nom_fichier)
    return textes, noms_fichiers

# Fonction de clustering avec KMeans
def clusteriser_textes(textes, k=3):
    vecteur = TfidfVectorizer()
    tfidf = vecteur.fit_transform(textes)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf)

    etiquettes = kmeans.labels_

    print("\nRésultats du clustering :")
    for i, texte in enumerate(textes):
        print(f"Texte {i+1} -> Cluster {etiquettes[i]}")

    return etiquettes

# Fonction pour diviser un texte en phrases
def decouper_en_phrases(texte):
    phrases = re.split(r'(?<=[.!?]) +', texte)
    return phrases

# Fonction pour extraire la phrase la plus représentative (phrase clé)
def extraire_phrase_cle(texte):
    phrases = decouper_en_phrases(texte)
    if len(phrases) == 0:
        return ""

    vecteur = TfidfVectorizer()
    tfidf = vecteur.fit_transform(phrases)

    similarite = cosine_similarity(tfidf)
    scores = similarite.sum(axis=1)

    index_meilleur = np.argmax(scores)
    phrase_cle = phrases[index_meilleur]

    print("\nPhrase clé extraite :")
    print(phrase_cle.strip())

    return phrase_cle.strip()

def afficher_histogramme(compteur, n=10):
    mots_communs = compteur.most_common(n)
    mots = [mot for mot, freq in mots_communs]
    freqs = [freq for mot, freq in mots_communs]

    plt.figure(figsize=(10,5))
    plt.bar(mots, freqs, color='skyblue')
    plt.title("Fréquence des mots les plus courants")
    plt.xlabel("Mots")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def afficher_matrice_similarite(textes):
    vecteur = TfidfVectorizer()
    tfidf = vecteur.fit_transform(textes)

    similarite = cosine_similarity(tfidf)

    plt.figure(figsize=(6,6))
    plt.imshow(similarite, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Matrice de similarité entre textes")
    plt.xlabel("Textes")
    plt.ylabel("Textes")
    plt.show()



if __name__ == "__main__":
    chemin = "exemple.txt"
    texte = charger_texte(chemin)
    texte_nettoye = nettoyer_texte(texte)

    print("-" * 40)
    print("\nTexte original:\n", texte[:300])
    print("-" * 40)
    print("\nTexte nettoyé:\n", texte_nettoye[:300])

    print("-" * 40)
    compteur = afficher_frequence_mots(texte_nettoye)
    generer_wordcloud(compteur)

    print("-" * 40)
    termes_tfidf = extraire_termes_tfidf(texte_nettoye)

    print("-" * 40)
    entites = extraire_entites_nommees(texte)

    print("-" * 40)
    theme = classer_texte_par_theme(texte_nettoye, themes_mots_cles)

    print("-" * 40)
    dossier = "corpus"
    textes, noms = charger_corpus(dossier)
    if len(textes) >= 2:
        etiquettes = clusteriser_textes(textes, k=3)
        afficher_matrice_similarite(textes)
    else:
        print("Pas assez de textes dans le corpus pour le clustering.")

    print("-" * 40)
    phrase_cle = extraire_phrase_cle(texte)

    afficher_histogramme(compteur)



