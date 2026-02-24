# PDF Fractures Analysis System 
Le système permet :

- Extraction automatique du texte  
- Détection des sections Abstract et Conclusion  
- Classification automatique (région / type / localisation)  
- Résumé automatique extractif basé sur DrBERT (modèle biomédical local)  
- Stockage dans une base SQLite  
- Interface Web interactive via Streamlit  

## Architecture
PDF → Extraction texte → Classification → Résumé (DrBERT) → SQLite → Interface Web Streamlit

## Technologies utilisées

- Python 3.10+
- PyMuPDF
- HuggingFace Transformers
- DrBERT (modèle biomédical)
- SQLite
- Streamlit

## Installation 

pip install -r requirements.txt

## Construire la base à partir d’un dossier de PDFs

Placez vos fichiers PDF dans : data/pdfs/

LANCEZ : python -m scripts.build_db --pdf_dir data/pdfs --db outputs/fractures.sqlite --json outputs/fractures.json --summarize

Cela va :
- Construire outputs/fractures.sqlite
- Générer outputs/fractures.json
- Générer un rapport pdf_count_report.json 

## Tester un seul PDF (inférence)

LANCEZ : python -m scripts.infer --pdf chemin/vers/fichier.pdf --db outputs/fractures.sqlite --json outputs/infer_result.json --summarize

Résultat généré : outputs/infer_result.json

## Lancer l’application Web

LANCEZ : streamlit run scripts/streamlit_app.py
Puis ouvrir : http://localhost:8501


## Windows Note (OpenMP)
Sous certains environnements Windows, PyTorch peut générer une erreur d'exécution OpenMP dupliquée.

Ce problème est géré automatiquement par les scripts grâce à la configuration des variables d'environnement requises.

Si nécessaire, vous pouvez exécuter manuellement la commande suivante :

PowerShell :

$env:KMP_DUPLICATE_LIB_OK="TRUE"

$env:OMP_NUM_THREADS="1"

$env:MKL_NUM_THREADS="1"

