#EN
# 🧠 NLP Project - Automatic Topic Detection in French Texts

This project uses Natural Language Processing (NLP) to analyze a French text and automatically detect its main topic. It includes key steps such as text cleaning, word frequency analysis, TF-IDF, named entity recognition, keyword-based classification, clustering, and automatic summarization.

## 🚀 Features

- Text preprocessing and cleaning
- Frequency analysis and WordCloud
- TF-IDF scoring of important terms
- Named Entity Recognition (NER)
- Theme classification based on keywords
- Clustering of multiple texts (KMeans)
- Automatic extraction of a key sentence (summary)
- Visualization: histogram, similarity matrix
- (Optional) Streamlit interface for interactive analysis

## 📁 Project Structure

- `analyse_sujet.py`: Main Python script
- `corpus/`: Folder with multiple `.txt` files
- `exemple.txt`: Single example text
- `wordcloud.png`, `histogramme.png`, `similarite.png`: Generated graphics
- `Rapport_TP_Complet.docx`: Project report in French

## ⚙️ Requirements

- Python 3.7+
- spaCy, nltk, sklearn, matplotlib, wordcloud

## 📌 Run the project

```bash
python analyse_sujet.py
```

## 🧪 Optional: Launch the Streamlit app

```bash
streamlit run app.py
```

---

#FR
# 🧠 Projet NLP - Analyse automatique du sujet d’un texte en français

Ce projet utilise le traitement automatique du langage (NLP) pour analyser un texte en français et détecter automatiquement son thème principal. Il comprend plusieurs étapes comme le nettoyage du texte, l’analyse de fréquence, TF-IDF, reconnaissance des entités nommées, classification par mots-clés, clustering et résumé automatique.

## 🚀 Fonctionnalités

- Nettoyage du texte
- Analyse de fréquence et nuage de mots
- Score TF-IDF des termes importants
- Reconnaissance d'entités nommées (NER)
- Classification par thème (mots-clés)
- Regroupement de textes (KMeans)
- Extraction automatique d’une phrase clé (résumé)
- Visualisation : histogramme, matrice de similarité
- (Optionnel) Interface Streamlit pour l’analyse interactive

## 📁 Structure du projet

- `analyse_sujet.py` : Script principal
- `corpus/` : Dossier avec plusieurs fichiers `.txt`
- `exemple.txt` : Texte d'exemple
- `wordcloud.png`, `histogramme.png`, `similarite.png` : Graphiques générés
- `Rapport_TP_Complet.docx` : Rapport du projet

## ⚙️ Prérequis

- Python 3.7+
- spaCy, nltk, sklearn, matplotlib, wordcloud

## 📌 Exécuter le script

```bash
python analyse_sujet.py
```

## 🧪 Optionnel : Lancer l’application Streamlit

```bash
streamlit run app.py
```
