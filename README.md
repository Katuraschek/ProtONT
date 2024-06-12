# Installation mit Conda

Um ProtONT zu installieren, benötigen Sie Python 3.9 oder eine neuere Version auf Ihrem System.

## 1. Erstellen und Aktivieren einer Conda-Umgebung

Erstellen Sie eine Conda-Umgebung und aktivieren Sie sie:

```bash
conda create -n ProtONT
conda activate ProtONT
```

## 2. Mit Git installieren (empfohlen)

Klonen Sie dieses Repository mit git clone https://github.com/Katuraschek/ProtONT.git und navigieren Sie dorthin, um es in der neu erstellten Conda-Umgebung zu installieren (verwenden Sie -e für editierbar):

```bash
pip install -e .
```

Nach Updates auf GitHub können Sie die neueste Version des Codes durch Ausführen von git pull im geklonten Verzeichnis verwenden.

## 3. Ohne Git installieren

Installieren Sie direkt von GitHub mit:

```bash
pip install git+https://github.com/Katuraschek/ProtONT.git#egg=ProtONT
```

Sie können überprüfen, ob ProtONT installiert wurde, indem Sie conda list verwenden. Um die neuesten Änderungen von GitHub zu übernehmen, können Sie Folgendes ausführen:

```bash
pip install -U --force-reinstall git+https://github.com/Katuraschek/ProtONT.git#egg=ProtONT
```

Dies ist weniger bequem, da Sie jedes Mal das gesamte Paket aus der Conda-Eingabeaufforderung neu installieren müssen.

## 4. Verwendung

Sie können das Paket aus Ihrer Conda-Umgebung verwenden, z.B. um das Beispiel-Notebook demonotebook.ipynb auszuführen. Um Jupyter Notebooks auszuführen, können Sie Jupyter in Ihrer Umgebung installieren:

```bash
conda install jupyter
```

## Kontakt

Für Fragen und Anregungen wenden Sie sich bitte an Katharina.Juraschek@childrens.harvard.edu.

Bitte verwenden Sie den Abschnitt "Issues" für Fehlerberichte und Feature-Anfragen.
